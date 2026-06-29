import copy
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from openai import APIError

from src.models.base_rag import BaseRAG
from src.models.query_logic_dag import (
    QueryLogicDAGBuilder,
    QueryLogicDAG,
    SubproblemNode,
    DependencyEdge,
)
from src.models.verify_non_cyclicity import (
    verify_dag_and_topological_sort,
    DependencyGraphCycleError,
    build_partial_order_fallback,
)
from src.models.dag_topological_rank import (
    compute_topological_ranks_from_verification,
    attach_topological_ranks_to_dag_dict,
    TopologicalRankError,
)
from src.models.dag_rank_resolver import ParentConditionedRankResolver
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init


# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Notebook / experiment용 최소 진행 로그 전용 logger.
# 기존 src.models.logic_rag logger와 분리해서, 필요한 진행률 로그만 켤 수 있게 한다.
progress_logger = logging.getLogger("logicrag.progress")


# [추가] Few-shot 예시 상수 (논문 Section 3.2)
# decompose_query()의 프롬프트에서 참조
# subproblems 형식: {"id", "text"} → QueryLogicDAGBuilder 입력 형식에 맞춤
QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES = """
Example 1:
Question: "Who is the mayor of the capital of France?"
Subproblems:
[
  {"id": 0, "text": "What is the capital of France?"},
  {"id": 1, "text": "Who is the mayor of this capital city?"}
]

Example 2:
Question: "When was the director of 'Inception' born, and what award did the film win at the Oscars?"
Subproblems:
[
  {"id": 0, "text": "Who directed the film 'Inception'?"},
  {"id": 1, "text": "When was this director born?"},
  {"id": 2, "text": "What award did 'Inception' win at the Oscars?"}
]

Example 3:
Question: "What is the population of the country where the inventor of the telephone was born?"
Subproblems:
[
  {"id": 0, "text": "Who invented the telephone?"},
  {"id": 1, "text": "In which country was this inventor born?"},
  {"id": 2, "text": "What is the population of this country?"}
]

Example 4:
Question: "What is the tallest building in Tokyo?"
Subproblems:
[
  {"id": 0, "text": "What is the tallest building in Tokyo?"}
]
"""


class LogicRAG(BaseRAG):
    def __init__(
        self,
        corpus_path: str = None,
        cache_dir: str = "./cache",
        filter_repeats: bool = False,
    ):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)

        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats

        # Query decomposition 결과인 subproblems를 Query Logic DAG G=(V,E)로 변환하는 Builder.
        self.dag_builder = QueryLogicDAGBuilder()

        # 마지막으로 생성/사용된 Query Logic DAG를 평가/디버깅용으로 저장한다.
        self.last_query_logic_dag = None
        self.last_query_logic_dag_dict = None
        self.last_dependency_analysis = []
        self.last_retrieval_history = []

        # DAG verification / repair settings
        self.max_dag_repair_attempts = 1
        self.dag_cycle_policy = "raise"  # "raise" or "fallback"

        # Dynamic DAG Adaptation cost cap.
        # 모든 rank는 처리하되, inference 중 새 subproblem 추가 횟수만 제한한다.
        self.max_dynamic_adaptations = 3

        # Official LogicRAG-style inference controls.
        # - max_rounds=3 matches the common experimental budget.
        # - warm-up runs a question-level retrieval gate before DAG construction.
        # - early stop runs a can_answer gate after each processed rank.
        self.max_rounds: Optional[int] = 3
        self.enable_warm_up = True
        self.enable_early_stop = True
        self.final_answer_policy = "structured"
        self.last_final_answer_policy = ""
        self.last_final_answer_source = ""
        self.last_summary_completeness = ""
        self.last_exit_type = ""
        self.last_answer_status = None
        self.last_answer_failure_type = None

    @staticmethod
    def _format_elapsed(start_time: float) -> str:
        return f"{time.perf_counter() - start_time:.2f}s"

    def _log_progress(
        self,
        message: str,
        start_time: Optional[float] = None,
        **fields: Any,
    ) -> None:
        """
        Notebook/experiment용 최소 진행 로그.

        기존 상세 debug logger와 분리하기 위해 logicrag.progress logger만 사용한다.
        """
        if not progress_logger.isEnabledFor(logging.INFO):
            return

        parts = [message]

        if start_time is not None:
            parts.append(f"elapsed={self._format_elapsed(start_time)}")

        for key, value in fields.items():
            parts.append(f"{key}={value}")

        progress_logger.info(" | ".join(parts))

    def set_max_rounds(self, max_rounds: Optional[int]) -> None:
        """Set the maximum number of Stage 5 rank-retrieval rounds.

        None means no explicit round budget. Non-negative integers cap the
        total number of executed rank rounds, including dynamically appended
        ranks.
        """
        if max_rounds is None:
            self.max_rounds = None
            return

        self.max_rounds = max(0, int(max_rounds))

    def set_enable_warm_up(self, value: bool) -> None:
        """Enable or disable the question-level warm-up gate."""
        self.enable_warm_up = self._coerce_bool(value)

    def set_enable_early_stop(self, value: bool) -> None:
        """Enable or disable the rank-level early-stop gate."""
        self.enable_early_stop = self._coerce_bool(value)

    def set_final_answer_policy(self, policy: str) -> None:
        """Set the final answer routing policy."""
        if policy not in {"generate", "structured"}:
            raise ValueError(
                f"Invalid final_answer_policy={policy!r}. "
                "Expected 'generate' or 'structured'."
            )
        self.final_answer_policy = policy

    # ==================================================================
    # Query decomposition
    # ==================================================================

    def decompose_query(self, question: str) -> Dict[str, Any]:
        """
        Decompose the input query into subproblems using few-shot prompting.
        """
        try:
            prompt = f"""You are an expert at decomposing complex questions into smaller, logically ordered subproblems.

Given a question, you must:
1. Decompose the question into a minimal set of subproblems. Each subproblem must have an "id" (integer, starting from 0) and a "text" (the subproblem question string).
2. If the question is simple (single-hop, no decomposition needed), output a single subproblem identical to the original question.

Here are some examples:
{QUERY_DECOMPOSITION_FEW_SHOT_EXAMPLES}

Now decompose the following question:
Question: "{question}"

Please format your response as a JSON object with these keys:
- "subproblems": list of objects, each with "id" (int) and "text" (string)
- "is_simple": boolean

Respond ONLY with the JSON object, no additional text."""

            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")

            result = fix_json_response(response)

            if result is None:
                return {
                    "subproblems": [{"id": 0, "text": question}],
                    "is_simple": True,
                }

            if (
                "subproblems" not in result
                or not isinstance(result["subproblems"], list)
                or len(result["subproblems"]) == 0
            ):
                result["subproblems"] = [{"id": 0, "text": question}]

            if "is_simple" not in result:
                result["is_simple"] = len(result["subproblems"]) <= 1

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in decompose_query: {e}{Style.RESET_ALL}")
            return {
                "subproblems": [{"id": 0, "text": question}],
                "is_simple": True,
            }

    # ==================================================================
    # Small coercion helpers
    # ==================================================================

    @staticmethod
    def _as_int_or_none(value: Any) -> Optional[int]:
        """bool을 제외하고 int 변환 가능한 값만 int로 변환한다."""
        if isinstance(value, bool):
            return None

        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        """
        LLM이 true/false를 문자열로 반환해도 안전하게 bool로 변환한다.

        bool("false") == True 문제를 방지하기 위한 helper.
        """
        if isinstance(value, bool):
            return value

        if isinstance(value, str):
            normalized = value.strip().lower()

            if normalized in {"true", "yes", "y", "1"}:
                return True

            if normalized in {"false", "no", "n", "0", ""}:
                return False

        return bool(value)

    @staticmethod
    def _is_valid_final_subanswer_summary(summary: Any) -> bool:
        """Return whether a final_subanswer_summary has usable content."""
        if summary is None:
            return False

        if isinstance(summary, (list, dict)):
            return bool(summary)

        if isinstance(summary, str):
            stripped = summary.strip()
            if not stripped or stripped in {"[]", "{}"}:
                return False

            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return True

            if isinstance(parsed, (list, dict)):
                return bool(parsed)

            return parsed is not None

        return True

    # ==================================================================
    # Dynamic DAG Adaptation helpers
    # ==================================================================

    def _match_node_id_by_text(
        self,
        dag: QueryLogicDAG,
        text: str,
    ) -> Optional[int]:
        """
        DAG의 V에서 주어진 text와 일치하는 node id를 반환한다.
        대소문자와 양끝 공백은 무시한다.
        """
        target = (text or "").strip().lower()
        if not target:
            return None

        for node_id, node in dag.V.items():
            if node.text.strip().lower() == target:
                return node_id

        return None

    def _maybe_add_subproblem(
        self,
        question: str,
        info_summary: str,
        dag: QueryLogicDAG,
        sub_answers: Dict[int, str],
        current_max_rank: int,
        unresolved_answers: Optional[List[Dict[str, Any]]] = None,
        current_rank: Optional[int] = None,
        current_nodes: Optional[List[Dict[str, Any]]] = None,
        rank_result: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Dynamic DAG Adaptation hook.

        ParentConditionedRankResolver.run()의 매 rank 처리 직후 호출된다.
        unresolved / insufficient answer가 있을 때 LLM에게 새 subproblem 추가 여부를 묻고,
        필요하면 QueryLogicDAG 객체에 node와 edge를 추가한다.
        """
        unresolved_answers = unresolved_answers or []
        current_nodes = current_nodes or []
        rank_result = rank_result or {}

        # ── Step 1: LLM에게 보낼 요약 정보 구성 ──
        nodes_summary_lines = []
        for node_id in sorted(dag.V.keys()):
            node_text = dag.V[node_id].text
            answer = sub_answers.get(node_id, "(unresolved)")
            nodes_summary_lines.append(
                f'- id={node_id}, text="{node_text}", answer="{answer}"'
            )
        nodes_summary = "\n".join(nodes_summary_lines)

        current_nodes_payload = []
        for node in current_nodes:
            if not isinstance(node, dict):
                continue

            current_nodes_payload.append({
                "node_id": node.get("node_id"),
                "subproblem": node.get("subproblem", ""),
            })

        unresolved_payload = []
        for answer in unresolved_answers:
            if not isinstance(answer, dict):
                continue

            unresolved_payload.append({
                "node_id": answer.get("node_id"),
                "subproblem": answer.get("subproblem", ""),
                "answer": answer.get("answer", ""),
                "is_answered": answer.get("is_answered", False),
                "missing_info": answer.get("missing_info", ""),
                "evidence_summary": answer.get("evidence_summary", ""),
            })

        rank_result_payload = json.dumps(rank_result, ensure_ascii=False, indent=2)

        prompt = f"""You are performing Dynamic DAG Adaptation for LogicRAG.

Original question Q:
{question}

Current topological rank just processed:
{current_rank}

Current rank subproblems:
{json.dumps(current_nodes_payload, ensure_ascii=False, indent=2)}

Existing subproblems in the DAG with their resolved answers, if any:
{nodes_summary}

Unresolved or insufficiently answered subproblems from the current rank:
{json.dumps(unresolved_payload, ensure_ascii=False, indent=2)}

Current rolling memory after resolving the current rank:
{info_summary}

Full current rank result:
{rank_result_payload}

Task:
Decide whether ONE additional subproblem should be added to the Query Logic Dependency Graph.

Paper-grounded trigger:
- Add a new subproblem only when the current retrieval/resolution exposes insufficient context, an unresolved dependency, or a missing intermediate fact needed to resolve the original question.
- Prefer using the unresolved/missing_info fields as the main evidence for adding a new subproblem.

Rules:
- If there is no unresolved or missing intermediate information, output {{"need_new_subproblem": false}} unless the original question still clearly requires one missing intermediate subproblem.
- If a new subproblem is needed, write a concrete, answerable retrieval subproblem.
- The new subproblem MUST NOT duplicate any existing subproblem.
- depends_on must reference existing subproblem ids only, or be an empty list if independent.
- Use depends_on to point to the existing subproblem answers that the new subproblem logically depends on.
- Add at most ONE subproblem per call.
- Do not answer the new subproblem.
- Return ONLY a JSON object.

Output schema:
{{
  "need_new_subproblem": boolean,
  "new_subproblem_text": string or null,
  "depends_on": [list of integers],
  "reason": string
}}"""

        # ── Step 2: LLM 호출 + JSON 파싱 ──
        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)
        except Exception as e:
            logger.error(
                f"{Fore.RED}Error in _maybe_add_subproblem LLM call: {e}{Style.RESET_ALL}"
            )
            return None

        if not isinstance(result, dict):
            logger.warning(
                f"{Fore.YELLOW}Dynamic adaptation: LLM response was not a dict.{Style.RESET_ALL}"
            )
            return None

        if not self._coerce_bool(result.get("need_new_subproblem", False)):
            logger.info(
                f"{Fore.GREEN}Dynamic adaptation: LLM determined no new subproblem is needed.{Style.RESET_ALL}"
            )
            return None

        # ── Step 3: 응답 검증 ──
        new_text = result.get("new_subproblem_text") or ""
        if not isinstance(new_text, str):
            return None

        new_text = new_text.strip()
        if not new_text:
            logger.warning(
                f"{Fore.YELLOW}Dynamic adaptation: empty new_subproblem_text.{Style.RESET_ALL}"
            )
            return None

        if self._match_node_id_by_text(dag, new_text) is not None:
            logger.warning(
                f"{Fore.YELLOW}Dynamic adaptation: new subproblem duplicates an existing one. Skip.{Style.RESET_ALL}"
            )
            return None

        existing_ids = set(dag.V.keys())
        depends_on_raw = result.get("depends_on", []) or []
        depends_on: List[int] = []

        if isinstance(depends_on_raw, list):
            for raw_id in depends_on_raw:
                pid = self._as_int_or_none(raw_id)
                if pid is None:
                    continue

                if pid in existing_ids and pid not in depends_on:
                    depends_on.append(pid)

        reason = str(result.get("reason", "") or "Dynamic adaptation.").strip()

        # ── Step 4: DAG mutation (node 추가) ──
        new_id = max(dag.V.keys()) + 1 if dag.V else 0

        try:
            dag.add_node(
                SubproblemNode(
                    id=new_id,
                    text=new_text,
                    metadata={
                        "source": "dynamic_adaptation",
                        "created_by": "maybe_add_subproblem",
                        "added_after_rank": current_rank,
                        "added_at_round_max_rank": current_max_rank,
                        "trigger_unresolved_answers": unresolved_payload,
                    },
                )
            )
        except ValueError as e:
            logger.error(
                f"{Fore.RED}Dynamic adaptation: failed to add node {new_id}: {e}{Style.RESET_ALL}"
            )
            return None

        # ── Step 5: DAG mutation (edge 추가) ──
        for parent_id in depends_on:
            try:
                dag.add_edge(
                    DependencyEdge(
                        prerequisite_id=parent_id,
                        dependent_id=new_id,
                        reason=reason,
                        metadata={
                            "source": "dynamic_adaptation",
                            "relation_type": "logical_precedence",
                            "created_by": "maybe_add_subproblem",
                        },
                    )
                )
            except ValueError as e:
                logger.warning(
                    f"{Fore.YELLOW}Dynamic adaptation: failed to add edge "
                    f"{parent_id}->{new_id}: {e}{Style.RESET_ALL}"
                )

        # ── Step 6: 새 노드의 rank 계산 ──
        # 논문 Algorithm 1 line 16: 새 subproblem을 current rank sequence 뒤에 append한다.
        # 따라서 parent rank보다 뒤이면서도 현재 처리 sequence의 마지막 rank보다 뒤여야 한다.
        if depends_on:
            parent_ranks = [int(dag.ranks.get(parent_id, 0)) for parent_id in depends_on]
            dependency_safe_rank = max(parent_ranks) + 1
        else:
            dependency_safe_rank = 0

        new_rank = max(int(current_max_rank) + 1, dependency_safe_rank)
        dag.ranks[new_id] = new_rank

        logger.info(
            f"{Fore.YELLOW}Dynamic adaptation: added subproblem #{new_id} "
            f"\"{new_text}\" at rank {new_rank} "
            f"(depends_on={depends_on}). Reason: {reason}{Style.RESET_ALL}"
        )

        return {
            "new_subproblem_id": new_id,
            "new_subproblem_text": new_text,
            "new_rank": new_rank,
            "depends_on": depends_on,
            "reason": reason,
            "trigger_unresolved_answers": unresolved_payload,
        }

    # ==================================================================
    # Retrieval wrapper used by ParentConditionedRankResolver
    # ==================================================================

    def _retrieve_for_query(
        self,
        query: str,
        retrieved_chunks_set: Optional[set] = None,
    ) -> List[str]:
        """
        Resolver가 unified query 하나를 실제 retrieval에 넘길 때 사용하는 wrapper.
        """
        if self.filter_repeats and retrieved_chunks_set is not None:
            contexts = self._retrieve_with_filter(query, retrieved_chunks_set)
            for chunk in contexts:
                retrieved_chunks_set.add(chunk)
            return contexts

        return self.retrieve(query)

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> List[str]:
        """
        filter_repeats=True일 때 이미 사용한 context chunk를 제외하고 retrieval한다.
        """
        if retrieved_chunks_set is None:
            retrieved_chunks_set = set()

        if self.corpus_embeddings is None or not self.corpus:
            return []

        target_k = min(int(self.top_k), len(self.corpus))
        if target_k <= 0:
            return []

        unique_results = []
        retrieval_window = target_k

        while len(unique_results) < target_k and retrieval_window <= len(self.corpus):
            all_results = self._retrieve_top_n(query, retrieval_window)
            unique_results = [
                chunk
                for chunk in all_results
                if chunk not in retrieved_chunks_set
            ]

            if len(unique_results) >= target_k:
                break

            retrieval_window += target_k

        return unique_results[:target_k]

    def _retrieve_top_n(self, query: str, n: int) -> List[str]:
        """
        top_k를 임시로 바꿔 top-n retrieval을 수행한다.
        """
        old_top_k = self.top_k

        try:
            self.top_k = min(int(n), len(self.corpus))
            return self.retrieve(query)
        finally:
            self.top_k = old_top_k

    # ==================================================================
    # Warm-up / answerability helpers
    # ==================================================================

    def refine_summary_with_context(
        self,
        question: str,
        new_contexts: List[str],
        current_summary: str = "",
    ) -> str:
        """Summarize retrieved contexts into the rolling information summary.

        This is used by the official LogicRAG-style warm-up gate. The summary
        is also passed as Stage 5 initial_memory when warm-up cannot answer.
        """
        context_text = "\n\n".join(new_contexts or []).strip()
        current_summary = (current_summary or "").strip()

        if not context_text and not current_summary:
            return ""

        try:
            if current_summary:
                prompt = f"""Please refine the current information summary using newly retrieved information.

Question:
{question}

Current summary:
{current_summary}

New retrieved information:
{context_text}

Rules:
- Integrate only facts that may help answer the question.
- Remove redundancy and irrelevant details.
- Preserve exact names, dates, numbers, relations, and comparison targets.
- Do not invent unsupported facts.

Refined summary:
"""
            else:
                prompt = f"""Please create a concise information summary from the retrieved documents.

Question:
{question}

Retrieved information:
{context_text}

Rules:
- Include only facts that may help answer the question.
- Exclude irrelevant details.
- Preserve exact names, dates, numbers, relations, and comparison targets.
- Do not invent unsupported facts.

Summary:
"""

            return get_response_with_retry(prompt).strip()

        except Exception as e:
            logger.error(f"{Fore.RED}Error refining summary with context: {e}{Style.RESET_ALL}")
            if current_summary and context_text:
                return f"{current_summary}\n\nNew retrieved information:\n{context_text}".strip()
            return current_summary or context_text

    def warm_up_analysis(self, question: str, info_summary: str) -> Dict[str, Any]:
        """Question-level answerability gate before DAG construction.

        can_answer=True means the system can return immediately using only the
        warm-up information summary. Parse failures deliberately fall back to
        can_answer=False to avoid unsafe premature exits.
        """
        info_summary = (info_summary or "").strip()

        if not info_summary:
            return {
                "can_answer": False,
                "missing_info": "No warm-up information was retrieved.",
                "subquery": question,
                "current_understanding": "",
                "dependencies": [],
                "missing_reason": "empty_warm_up_summary",
            }

        try:
            prompt = f"""You are performing warm-up analysis before LogicRAG DAG reasoning.

Original question:
{question}

Available information summary:
{info_summary}

Task:
Decide whether the original question can be answered completely using ONLY the available information summary.

Return can_answer=true ONLY if:
- The final answer to the original question is directly supported by the summary, or
- The final answer can be produced by simple comparison/composition of facts already present in the summary.
- No required fact is missing.
- No additional retrieval is needed.

If any required fact is missing, ambiguous, conflicting, or not directly supported, return can_answer=false.

Also provide dependencies: the key information needs that deeper DAG reasoning should resolve if can_answer is false.

Return ONLY a JSON object with this schema:
{{
  "can_answer": boolean,
  "missing_info": string,
  "subquery": string,
  "current_understanding": string,
  "dependencies": [list of strings],
  "missing_reason": string
}}
"""

            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            result = fix_json_response(response)

            if not isinstance(result, dict):
                raise ValueError("warm-up response is not a JSON object")

            result["can_answer"] = self._coerce_bool(result.get("can_answer", False))

            dependencies = result.get("dependencies", [])
            if not isinstance(dependencies, list):
                dependencies = []

            result["dependencies"] = [
                str(dep).strip()
                for dep in dependencies
                if str(dep).strip()
            ]
            result.setdefault("missing_info", "")
            result.setdefault("subquery", question)
            result.setdefault("current_understanding", "")
            result.setdefault("missing_reason", "")

            return result

        except Exception as e:
            logger.error(f"{Fore.RED}Error in warm_up_analysis: {e}{Style.RESET_ALL}")
            return {
                "can_answer": False,
                "missing_info": f"Warm-up analysis failed: {e}",
                "subquery": question,
                "current_understanding": "",
                "dependencies": [],
                "missing_reason": "warm_up_analysis_error",
            }

    def generate_answer(self, question: str, info_summary: str) -> str:
        """Generate a direct final answer from the current information summary.

        Used for warm-up early return, rank-level early stop, and max_rounds
        budget exhaustion.
        """
        prompt = f"""You must give ONLY the direct final answer in the most concise way possible.

Question:
{question}

Information summary:
{info_summary}

Rules:
- Use ONLY the information summary.
- Do not explain.
- Do not include reasoning steps.
- Do not include citations.
- If the answer is yes/no, answer only "Yes." or "No."
- If the answer is a name, date, number, or short phrase, return only that value.
- Do not invent unsupported facts.

Final answer:
"""
        try:
            result = get_response_with_retry(prompt).strip()

        except APIError as e:
            logger.error(f"{Fore.RED}APIError in generate_answer: {e}{Style.RESET_ALL}")
            self.last_answer_status = "failed"
            self.last_answer_failure_type = "api_error"
            return ""

        if result:
            self.last_answer_status = "ok"
            self.last_answer_failure_type = None
        else:
            logger.warning(f"{Fore.YELLOW}Empty content in generate_answer.{Style.RESET_ALL}")
            self.last_answer_status = "failed"
            self.last_answer_failure_type = "empty_content"

        return result

    # ==================================================================
    # Final composition
    # ==================================================================

    def compose_final_answer(self, question: str, subanswer_summary: str) -> str:
        """
        논문 Algorithm 1의 마지막 단계 Compose({a_i})를 수행한다.
        """
        prompt = f"""You must compose the final answer using ONLY the intermediate subproblem answers.

Original question:
{question}

Intermediate subproblem answers in topological order:
{subanswer_summary}

Rules:
- Give ONLY the direct final answer.
- Do not explain.
- Do not include reasoning steps.
- Do not include citations.
- If the answer is a simple yes/no, just say "Yes." or "No."
- If the answer is a name, date, number, or short phrase, return only that value.
- Do not invent facts not supported by the intermediate answers.

Final answer:
"""
        try:
            result = get_response_with_retry(prompt).strip()

        except APIError as e:
            logger.error(f"{Fore.RED}APIError in compose_final_answer: {e}{Style.RESET_ALL}")
            self.last_answer_status = "failed"
            self.last_answer_failure_type = "api_error"
            return ""

        if result:
            self.last_answer_status = "ok"
            self.last_answer_failure_type = None
        else:
            logger.warning(f"{Fore.YELLOW}Empty content in compose_final_answer.{Style.RESET_ALL}")
            self.last_answer_status = "failed"
            self.last_answer_failure_type = "empty_content"

        return result

    # ==================================================================
    # DAG verification / repair helpers
    # ==================================================================

    @staticmethod
    def _get_dag_node_edge_keys(dag_dict: Dict[str, Any]) -> Tuple[str, str]:
        """DAG dict에서 node field와 edge field 이름을 찾는다."""
        if "nodes" in dag_dict and "edges" in dag_dict:
            return "nodes", "edges"

        if "V" in dag_dict and "E" in dag_dict:
            return "V", "E"

        raise ValueError("DAG dict must have either nodes/edges or V/E.")

    @staticmethod
    def _nodes_payload_from_dag_dict(dag_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
        """LLM repair prompt용 node 리스트를 만든다."""
        node_key, _ = LogicRAG._get_dag_node_edge_keys(dag_dict)
        raw_nodes = dag_dict[node_key]

        nodes_payload = []

        for raw_key, raw_node in raw_nodes.items():
            if not isinstance(raw_node, dict):
                continue

            node_id = LogicRAG._as_int_or_none(raw_node.get("id", raw_key))
            if node_id is None:
                continue

            text = raw_node.get("text", "")

            nodes_payload.append({
                "id": node_id,
                "text": text,
            })

        return sorted(nodes_payload, key=lambda item: item["id"])

    @staticmethod
    def _rebuild_dag_indexes_dict(dag_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        LLM이 edge를 수정한 뒤 parents / children index를 다시 계산한다.
        """
        node_key, edge_key = LogicRAG._get_dag_node_edge_keys(dag_dict)

        raw_nodes = dag_dict[node_key]
        raw_edges = dag_dict[edge_key]

        node_ids = []

        for raw_key, raw_node in raw_nodes.items():
            if not isinstance(raw_node, dict):
                continue

            node_id = LogicRAG._as_int_or_none(raw_node.get("id", raw_key))
            if node_id is not None:
                node_ids.append(node_id)

        node_id_set = set(node_ids)

        parents = {node_id: set() for node_id in node_ids}
        children = {node_id: set() for node_id in node_ids}

        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue

            pre = LogicRAG._as_int_or_none(edge.get("prerequisite_id"))
            dep = LogicRAG._as_int_or_none(edge.get("dependent_id"))

            if pre is None or dep is None:
                continue

            if pre not in node_id_set or dep not in node_id_set:
                continue

            if pre == dep:
                continue

            children[pre].add(dep)
            parents[dep].add(pre)

        dag_dict["parents"] = {
            node_id: sorted(list(parent_ids))
            for node_id, parent_ids in parents.items()
        }

        dag_dict["children"] = {
            node_id: sorted(list(child_ids))
            for node_id, child_ids in children.items()
        }

        return dag_dict

    @staticmethod
    def _query_logic_dag_from_verified_result(
        dag_dict: Dict[str, Any],
        verification_result: Dict[str, Any],
        rank_result: Dict[str, Any],
    ) -> QueryLogicDAG:
        """
        verified/repaired DAG dict와 verification result를 runtime QueryLogicDAG로 재구성한다.

        이유:
        - Dynamic DAG Adaptation hook은 QueryLogicDAG 객체를 mutate한다.
        - cycle repair가 발생하면 원본 dag 객체의 edge set과 verified DAG edge set이 달라질 수 있다.
        - 따라서 Stage 5에는 verified DAG 기준 runtime object를 넘겨야 한다.
        """
        node_key, edge_key = LogicRAG._get_dag_node_edge_keys(dag_dict)

        raw_nodes = dag_dict.get(node_key, {}) or {}
        raw_edges = dag_dict.get(edge_key, []) or []

        nodes: Dict[int, SubproblemNode] = {}

        for raw_key, raw_node in raw_nodes.items():
            if not isinstance(raw_node, dict):
                continue

            node_id = LogicRAG._as_int_or_none(raw_node.get("id", raw_key))
            if node_id is None:
                continue

            text = raw_node.get("text", "")
            if not isinstance(text, str) or not text.strip():
                continue

            metadata = raw_node.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            nodes[node_id] = SubproblemNode(
                id=node_id,
                text=text.strip(),
                metadata=metadata,
            )

        raw_edge_by_key: Dict[Tuple[int, int], Dict[str, Any]] = {}
        for raw_edge in raw_edges:
            if not isinstance(raw_edge, dict):
                continue

            pre = LogicRAG._as_int_or_none(raw_edge.get("prerequisite_id"))
            dep = LogicRAG._as_int_or_none(raw_edge.get("dependent_id"))

            if pre is None or dep is None:
                continue

            raw_edge_by_key[(pre, dep)] = raw_edge

        verified_edges = verification_result.get("valid_dependency_edges", []) or []

        edges: List[DependencyEdge] = []
        seen_edges = set()

        for edge in verified_edges:
            if not isinstance(edge, dict):
                continue

            pre = LogicRAG._as_int_or_none(edge.get("prerequisite_id"))
            dep = LogicRAG._as_int_or_none(edge.get("dependent_id"))

            if pre is None or dep is None:
                continue

            if pre not in nodes or dep not in nodes:
                continue

            if pre == dep:
                continue

            edge_key_tuple = (pre, dep)
            if edge_key_tuple in seen_edges:
                continue

            seen_edges.add(edge_key_tuple)

            original_edge = raw_edge_by_key.get(edge_key_tuple, {})
            metadata = original_edge.get("metadata", {})
            if not isinstance(metadata, dict):
                metadata = {}

            reason = (
                edge.get("reason")
                or original_edge.get("reason")
                or ""
            )

            edges.append(
                DependencyEdge(
                    prerequisite_id=pre,
                    dependent_id=dep,
                    reason=str(reason),
                    metadata=metadata,
                )
            )

        metadata = dag_dict.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        runtime_dag = QueryLogicDAG(
            V=nodes,
            E=edges,
            metadata=metadata,
        )
        runtime_dag.rebuild_indexes()

        sorted_node_ids = []
        for raw_id in verification_result.get("sorted_node_ids", []) or []:
            node_id = LogicRAG._as_int_or_none(raw_id)
            if node_id is not None and node_id in runtime_dag.V:
                sorted_node_ids.append(node_id)

        runtime_dag.set_topological_order(sorted_node_ids)
        runtime_dag.mark_acyclic_verified(bool(verification_result.get("is_dag", False)))

        ranks: Dict[int, int] = {}
        for raw_id, raw_rank in (rank_result.get("ranks", {}) or {}).items():
            node_id = LogicRAG._as_int_or_none(raw_id)
            rank = LogicRAG._as_int_or_none(raw_rank)

            if node_id is None or rank is None:
                continue

            if node_id in runtime_dag.V:
                ranks[node_id] = rank

        runtime_dag.set_ranks(ranks)

        return runtime_dag

    def _repair_cyclic_dag_with_llm(
        self,
        question: str,
        dag_dict: Dict[str, Any],
        dag_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Cycle이 있는 DAG의 edge set을 LLM으로 repair한다."""
        _, edge_key = self._get_dag_node_edge_keys(dag_dict)

        nodes_payload = self._nodes_payload_from_dag_dict(dag_dict)
        current_edges = dag_result.get("valid_dependency_edges") or dag_dict.get(edge_key, [])

        cycle_node_ids = dag_result.get("cycle_node_ids", [])
        cycle_dependencies = dag_result.get("cycle_dependencies", [])
        blocked_node_ids = dag_result.get("blocked_node_ids", [])

        prompt = f"""
You are repairing the edge set of a Query Logic Dependency Graph.

Original question:
{question}

Subproblem nodes:
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Current directed edges:
{json.dumps(current_edges, ensure_ascii=False, indent=2)}

Detected cycle node ids:
{json.dumps(cycle_node_ids, ensure_ascii=False)}

Detected cycle subproblems:
{json.dumps(cycle_dependencies, ensure_ascii=False, indent=2)}

Blocked node ids:
{json.dumps(blocked_node_ids, ensure_ascii=False)}

Task:
Repair the edge set so that the graph becomes a valid DAG.

Rules:
- Use only node ids from the provided subproblem nodes.
- Edge direction must be prerequisite_id -> dependent_id.
- Do not create self-loops.
- Remove or modify the minimum number of edges needed to break cycles.
- Preserve necessary logical dependencies when possible.
- Do not add redundant transitive edges.
- Each edge must include a short reason.
- Return ONLY a JSON object.

Output schema:
{{
  "edges": [
    {{
      "prerequisite_id": integer,
      "dependent_id": integer,
      "reason": string
    }}
  ]
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")

            repaired = fix_json_response(response)

            if not isinstance(repaired, dict) or not isinstance(repaired.get("edges"), list):
                logger.error(
                    f"{Fore.RED}Failed to parse repaired DAG edges. "
                    f"Using original DAG for re-verification.{Style.RESET_ALL}"
                )
                return copy.deepcopy(dag_dict)

            repaired_edges = []

            for edge in repaired["edges"]:
                if not isinstance(edge, dict):
                    continue

                pre = self._as_int_or_none(edge.get("prerequisite_id"))
                dep = self._as_int_or_none(edge.get("dependent_id"))

                if pre is None or dep is None:
                    continue

                repaired_edges.append({
                    "prerequisite_id": pre,
                    "dependent_id": dep,
                    "reason": edge.get("reason", ""),
                })

            repaired_dag_dict = copy.deepcopy(dag_dict)
            repaired_dag_dict[edge_key] = repaired_edges
            repaired_dag_dict = self._rebuild_dag_indexes_dict(repaired_dag_dict)

            return repaired_dag_dict

        except Exception as e:
            logger.error(f"{Fore.RED}Error during DAG repair: {e}{Style.RESET_ALL}")
            return copy.deepcopy(dag_dict)

    def _verify_sort_dependencies_with_repair(
        self,
        question: str,
        dag_dict: Dict[str, Any],
        max_repair_attempts: int = 1,
        on_repair_failure: str = "raise",
    ) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
        """
        DAG 검증 + topological sort + cycle repair orchestration.
        """
        if on_repair_failure not in {"raise", "fallback"}:
            raise ValueError("on_repair_failure must be either 'raise' or 'fallback'.")

        verification_history = []
        current_dag_dict = copy.deepcopy(dag_dict)

        dag_result = verify_dag_and_topological_sort(current_dag_dict)

        verification_history.append({
            "attempt": 0,
            "type": "initial_verification",
            "dag": current_dag_dict,
            "dag_verification": dag_result,
        })

        if dag_result["is_dag"]:
            return dag_result["sorted_dependencies"], current_dag_dict, verification_history

        if not dag_result.get("has_cycle", False):
            if on_repair_failure == "fallback":
                fallback_dependencies = build_partial_order_fallback(dag_result)
                return fallback_dependencies, current_dag_dict, verification_history

            raise ValueError(
                f"Invalid DAG input. invalid_edges={dag_result.get('invalid_edges', [])}"
            )

        for attempt in range(1, max_repair_attempts + 1):
            logger.warning(
                f"{Fore.YELLOW}Cycle detected in Query Logic DAG. "
                f"Attempting LLM repair {attempt}/{max_repair_attempts}.{Style.RESET_ALL}"
            )

            repaired_dag_dict = self._repair_cyclic_dag_with_llm(
                question=question,
                dag_dict=current_dag_dict,
                dag_result=dag_result,
            )

            repaired_result = verify_dag_and_topological_sort(repaired_dag_dict)

            verification_history.append({
                "attempt": attempt,
                "type": "llm_repair_verification",
                "dag": repaired_dag_dict,
                "dag_verification": repaired_result,
            })

            current_dag_dict = repaired_dag_dict
            dag_result = repaired_result

            if dag_result["is_dag"]:
                logger.info(f"{Fore.GREEN}DAG repair succeeded.{Style.RESET_ALL}")
                return dag_result["sorted_dependencies"], current_dag_dict, verification_history

            if not dag_result.get("has_cycle", False):
                break

        if on_repair_failure == "fallback":
            logger.warning(
                f"{Fore.YELLOW}DAG repair failed. Using partial-order fallback.{Style.RESET_ALL}"
            )
            fallback_dependencies = build_partial_order_fallback(dag_result)
            return fallback_dependencies, current_dag_dict, verification_history

        raise DependencyGraphCycleError(dag_result)

    # ==================================================================
    # Main entrypoint
    # ==================================================================

    def answer_question(
        self,
        question: str,
        max_rounds: Optional[int] = None,
        enable_warm_up: Optional[bool] = None,
        enable_early_stop: Optional[bool] = None,
    ) -> Tuple[str, List[str], int]:
        """
        LogicRAG main pipeline with Dynamic DAG Adaptation.

        Stage 0: official LogicRAG-style warm-up retrieval gate
        Stage 1: query decomposition
        Stage 2: Query Logic DAG construction
        Stage 3: DAG verification + cycle repair
        Stage 4: topological rank calculation
        Stage 5: parent-answer conditioned rank resolution + Dynamic DAG Adaptation
        Stage 6: final answer composition
        """
        total_start_time = time.perf_counter()

        dependency_analysis_history = []
        retrieved_chunks_set = set() if self.filter_repeats else None

        effective_max_rounds = self.max_rounds if max_rounds is None else max(0, int(max_rounds))
        effective_enable_warm_up = (
            self.enable_warm_up if enable_warm_up is None else bool(enable_warm_up)
        )
        effective_enable_early_stop = (
            self.enable_early_stop if enable_early_stop is None else bool(enable_early_stop)
        )
        effective_policy = self.final_answer_policy
        if effective_policy not in {"generate", "structured"}:
            raise ValueError(
                f"Invalid final_answer_policy={effective_policy!r}. "
                "Expected 'generate' or 'structured'."
            )

        self.last_final_answer_policy = effective_policy
        self.last_final_answer_source = ""
        self.last_summary_completeness = ""
        self.last_exit_type = ""
        self.last_answer_status = None
        self.last_answer_failure_type = None

        initial_memory = ""
        last_contexts: List[str] = []
        warm_up_result: Optional[Dict[str, Any]] = None
        warm_up_retrieval_record: Optional[Dict[str, Any]] = None

        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")

        self._log_progress(
            "[START] LogicRAG answer_question",
            question=str(question)[:120],
            corpus_docs=len(self.corpus),
            top_k=self.top_k,
            filter_repeats=self.filter_repeats,
            max_rounds=effective_max_rounds,
            warm_up=effective_enable_warm_up,
            early_stop=effective_enable_early_stop,
            final_answer_policy=effective_policy,
        )

        # ===============================================
        # == Stage 0: warm-up retrieval gate ==
        # 공식 LogicRAG 스타일: 원 질문으로 먼저 검색하고,
        # can_answer=True이면 DAG를 만들지 않고 즉시 반환한다.
        # can_answer=False이면 warm-up summary를 Stage 5 initial_memory로 넘긴다.
        # ===============================================
        if effective_enable_warm_up:
            stage_start_time = time.perf_counter()
            self._log_progress("[Stage 0/6] Warm-up retrieval START")

            warm_contexts = self._retrieve_for_query(
                question,
                retrieved_chunks_set=retrieved_chunks_set,
            )
            last_contexts = warm_contexts

            initial_memory = self.refine_summary_with_context(
                question=question,
                new_contexts=warm_contexts,
                current_summary="",
            )

            warm_up_result = self.warm_up_analysis(
                question=question,
                info_summary=initial_memory,
            )

            warm_up_retrieval_record = {
                "round": 0,
                "stage": "warm_up",
                "query": question,
                "contexts": warm_contexts,
                "memory": initial_memory,
                "analysis": warm_up_result,
            }

            dependency_analysis_history.append({
                "stage": "warm_up",
                "contexts": warm_contexts,
                "initial_memory": initial_memory,
                "analysis": warm_up_result,
            })

            self._log_progress(
                "[Stage 0/6] Warm-up retrieval DONE",
                start_time=stage_start_time,
                can_answer=warm_up_result.get("can_answer", False),
                dependencies=len(warm_up_result.get("dependencies", []) or []),
            )

            if self._coerce_bool(warm_up_result.get("can_answer", False)):
                answer = self.generate_answer(
                    question=question,
                    info_summary=initial_memory,
                )
                self.last_final_answer_source = "warmup_generate"
                self.last_summary_completeness = "none"
                self.last_exit_type = "warmup_exit"

                self._log_progress(
                    "[DONE] LogicRAG answer_question",
                    start_time=total_start_time,
                    rounds=0,
                    warm_up_early_return=True,
                )

                self.last_dependency_analysis = dependency_analysis_history
                self.last_retrieval_history = [warm_up_retrieval_record]

                return answer, last_contexts, 0

        # ===============================================
        # == Stage 1: query decomposition ==
        # 논문 Algorithm 1 line 1: decompose Q into subproblems P
        # ===============================================
        stage_start_time = time.perf_counter()
        self._log_progress("[Stage 1/6] Query decomposition START")

        decomposition = self.decompose_query(question)

        self._log_progress(
            "[Stage 1/6] Query decomposition DONE",
            start_time=stage_start_time,
            subproblems=len(decomposition.get("subproblems", []) or []),
            is_simple=decomposition.get("is_simple"),
        )

        logger.info(
            f"Query decomposition result: "
            f"{len(decomposition.get('subproblems', []))} subproblems detected."
        )
        logger.info(f"Subproblems: {decomposition.get('subproblems', [])}")

        # ===============================================
        # == Stage 2: Query Logic DAG construction ==
        # 논문 Algorithm 1 line 2-3: Initialize DAG, populate edges
        # ===============================================
        stage_start_time = time.perf_counter()
        self._log_progress("[Stage 2/6] Query Logic DAG construction START")

        dag = self.dag_builder.construct_from_subproblems(
            question=question,
            subproblems=decomposition["subproblems"],
        )

        self._log_progress(
            "[Stage 2/6] Query Logic DAG construction DONE",
            start_time=stage_start_time,
            nodes=len(dag.V),
            edges=len(dag.E),
        )

        self.last_query_logic_dag = dag
        dag_dict = dag.to_dict()

        dependency_analysis_history.append({
            "stage": "query_logic_dag_construction",
            "query_logic_dag": dag_dict,
        })

        logger.info(f"Constructed Query Logic DAG: {dag_dict}\n\n")

        # ===============================================
        # == Stage 3: DAG topological sort + cycle verification ==
        # ===============================================
        stage_start_time = time.perf_counter()
        self._log_progress("[Stage 3/6] DAG verification / repair START")

        sorted_dependencies, verified_dag_dict, dag_verification_history = (
            self._verify_sort_dependencies_with_repair(
                question=question,
                dag_dict=dag_dict,
                max_repair_attempts=self.max_dag_repair_attempts,
                on_repair_failure=self.dag_cycle_policy,
            )
        )

        # 보완: rank 계산 중 예외가 나더라도 이번 run의 verified/repaired DAG dict를 보존한다.
        self.last_query_logic_dag_dict = verified_dag_dict

        final_dag_result = dag_verification_history[-1]["dag_verification"]

        self._log_progress(
            "[Stage 3/6] DAG verification / repair DONE",
            start_time=stage_start_time,
            is_dag=final_dag_result.get("is_dag", False),
            has_cycle=final_dag_result.get("has_cycle", False),
            attempts=len(dag_verification_history),
        )

        if not final_dag_result.get("is_dag", False):
            raise DependencyGraphCycleError(final_dag_result)

        # ===============================================
        # == Stage 4: topological rank calculation ==
        # 논문 Algorithm 1 line 4: Topologically sort G to obtain ranks
        # ===============================================
        stage_start_time = time.perf_counter()
        self._log_progress("[Stage 4/6] Topological rank calculation START")

        try:
            topological_rank_result = compute_topological_ranks_from_verification(
                final_dag_result
            )

            verified_dag_dict = attach_topological_ranks_to_dag_dict(
                verified_dag_dict,
                topological_rank_result,
            )

            logger.info(f"Topological rank result: {topological_rank_result}\n\n")

            self._log_progress(
                "[Stage 4/6] Topological rank calculation DONE",
                start_time=stage_start_time,
                max_rank=topological_rank_result.get("max_rank", 0),
                rank_groups=len(topological_rank_result.get("rank_groups", {}) or {}),
            )

        except TopologicalRankError as e:
            logger.error(
                f"{Fore.RED}Failed to compute topological ranks: {e}{Style.RESET_ALL}"
            )
            raise

        runtime_dag = self._query_logic_dag_from_verified_result(
            dag_dict=verified_dag_dict,
            verification_result=final_dag_result,
            rank_result=topological_rank_result,
        )

        self.last_query_logic_dag = runtime_dag
        self.last_query_logic_dag_dict = verified_dag_dict

        dependency_analysis_history.append({
            "stage": "dag_verification_and_topological_ranking",
            "query_logic_dag": dag_dict,
            "verified_query_logic_dag": verified_dag_dict,
            "dag_verification_history": dag_verification_history,
            "topological_rank": topological_rank_result,
            "sorted_dependencies": sorted_dependencies,
        })

        logger.info(f"Verified Query Logic DAG: {verified_dag_dict}\n\n")
        logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")

        # ===============================================
        # == Stage 5: parent-answer conditioned retrieval +
        #             Dynamic DAG Adaptation ==
        # 논문 Algorithm 1 line 6-17:
        #   - line 6-13: rank 별 unified retrieval, sub 답 도출
        #   - line 14-17: 매 rank 후 Dynamic DAG Adaptation
        #
        # repair가 발생할 수 있으므로, 원본 dag가 아니라 verified DAG 기준으로
        # 재구성한 runtime_dag를 넘긴다.
        # ===============================================
        rank_resolver = ParentConditionedRankResolver(self)

        stage_start_time = time.perf_counter()
        self._log_progress(
            "[Stage 5/6] Parent-conditioned rank resolution START",
            max_dynamic_adaptations=self.max_dynamic_adaptations,
            max_rounds=effective_max_rounds,
            early_stop=effective_enable_early_stop,
            initial_memory_chars=len(initial_memory or ""),
        )

        stage5_result = rank_resolver.run(
            question=question,
            dag_result=final_dag_result,
            topological_rank_result=topological_rank_result,
            sorted_dependencies=sorted_dependencies,
            initial_memory=initial_memory,
            retrieved_chunks_set=retrieved_chunks_set,
            max_rounds=effective_max_rounds,
            dag=runtime_dag,
            max_dynamic_adaptations=self.max_dynamic_adaptations,
            enable_early_stop=effective_enable_early_stop,
        )

        last_contexts = stage5_result["last_contexts"]
        round_count = stage5_result["round_count"]
        retrieval_history = stage5_result["retrieval_history"]
        final_subanswer_summary = stage5_result["final_subanswer_summary"]

        self._log_progress(
            "[Stage 5/6] Parent-conditioned rank resolution DONE",
            start_time=stage_start_time,
            rounds=round_count,
            dynamic_adaptations=len(stage5_result.get("dynamic_adaptations", []) or []),
            early_stopped=stage5_result.get("early_stopped", False),
            hit_max_rounds=stage5_result.get("hit_max_rounds", False),
        )

        self.last_query_logic_dag = runtime_dag
        self.last_query_logic_dag_final_dict = runtime_dag.to_dict()

        dependency_analysis_history.append({
            "stage": "parent_answer_conditioned_rank_resolution_with_rolling_memory",
            "rank_groups": stage5_result["rank_groups"],
            "processed_rank_groups": stage5_result["processed_rank_groups"],
            "unprocessed_rank_groups": stage5_result.get("unprocessed_rank_groups", []),
            "resolved_answers_by_node_id": stage5_result["resolved_answers_by_node_id"],
            "final_memory": stage5_result["final_memory"],
            "final_subanswer_summary": final_subanswer_summary,
            "retrieval_history": retrieval_history,
            "dynamic_adaptations": stage5_result.get("dynamic_adaptations", []),
            "early_stopped": stage5_result.get("early_stopped", False),
            "early_stop_result": stage5_result.get("early_stop_result"),
            "hit_max_rounds": stage5_result.get("hit_max_rounds", False),
            "max_rounds": stage5_result.get("max_rounds"),
            "final_query_logic_dag": runtime_dag.to_dict(),
        })

        logger.info(
            f"Parent-answer conditioned rank resolution completed: "
            f"{round_count} rank rounds, "
            f"{len(stage5_result.get('dynamic_adaptations', []))} dynamic adaptations."
        )

        # ===============================================
        # == Stage 6: final answer composition ==
        # 논문 Algorithm 1 line 19: A = Compose({a_i})
        # ===============================================
        stage_start_time = time.perf_counter()
        self._log_progress("[Stage 6/6] Final answer composition START")

        final_memory = stage5_result.get("final_memory", "")
        early_stopped = stage5_result.get("early_stopped", False)
        hit_max_rounds = stage5_result.get("hit_max_rounds", False)
        has_valid_summary = self._is_valid_final_subanswer_summary(final_subanswer_summary)

        if effective_policy == "generate":
            if early_stopped:
                answer = self.generate_answer(
                    question=question,
                    info_summary=final_memory,
                )
                self.last_final_answer_source = "earlystop_generate"
                self.last_summary_completeness = "none"
                self.last_exit_type = "earlystop_exit"
            elif hit_max_rounds:
                answer = self.generate_answer(
                    question=question,
                    info_summary=final_memory,
                )
                self.last_final_answer_source = "maxround_generate"
                self.last_summary_completeness = "none"
                self.last_exit_type = "maxround_exit"
            else:
                answer = self.generate_answer(
                    question=question,
                    info_summary=final_memory,
                )
                self.last_final_answer_source = "normal_generate"
                self.last_summary_completeness = "none"
                self.last_exit_type = "normal_completion"
        elif early_stopped:
            early_stop_result = stage5_result.get("early_stop_result") or {}
            judgement = early_stop_result.get("judgement") or {}
            if not isinstance(judgement, dict):
                judgement = {}
            a15_final_answer = str(judgement.get("final_answer") or "").strip()

            if a15_final_answer:
                answer = a15_final_answer
                self.last_final_answer_source = "earlystop_a15"
                self.last_summary_completeness = "none"
                self.last_exit_type = "earlystop_exit"
                self.last_answer_status = "ok"
                self.last_answer_failure_type = None
            elif has_valid_summary:
                answer = self.compose_final_answer(
                    question=question,
                    subanswer_summary=final_subanswer_summary,
                )
                if answer and str(answer).strip():
                    self.last_final_answer_source = "earlystop_compose_fallback"
                    self.last_summary_completeness = "partial"
                    self.last_exit_type = "earlystop_exit"
                else:
                    answer = self.generate_answer(
                        question=question,
                        info_summary=final_memory,
                    )
                    self.last_final_answer_source = "earlystop_generate_fallback"
                    self.last_summary_completeness = "none"
                    self.last_exit_type = "earlystop_exit"
            else:
                answer = self.generate_answer(
                    question=question,
                    info_summary=final_memory,
                )
                self.last_final_answer_source = "earlystop_generate_fallback"
                self.last_summary_completeness = "none"
                self.last_exit_type = "earlystop_exit"
        elif hit_max_rounds:
            if has_valid_summary:
                answer = self.compose_final_answer(
                    question=question,
                    subanswer_summary=final_subanswer_summary,
                )
                if answer and str(answer).strip():
                    self.last_final_answer_source = "maxround_compose"
                    self.last_summary_completeness = "partial"
                    self.last_exit_type = "maxround_exit"
                else:
                    answer = self.generate_answer(
                        question=question,
                        info_summary=final_memory,
                    )
                    self.last_final_answer_source = "maxround_generate_fallback"
                    self.last_summary_completeness = "none"
                    self.last_exit_type = "maxround_exit"
            else:
                answer = self.generate_answer(
                    question=question,
                    info_summary=final_memory,
                )
                self.last_final_answer_source = "maxround_generate_fallback"
                self.last_summary_completeness = "none"
                self.last_exit_type = "maxround_exit"
        else:
            if has_valid_summary:
                answer = self.compose_final_answer(
                    question=question,
                    subanswer_summary=final_subanswer_summary,
                )
                if answer and str(answer).strip():
                    self.last_final_answer_source = "normal_compose"
                    self.last_summary_completeness = "complete"
                    self.last_exit_type = "normal_completion"
                else:
                    answer = self.generate_answer(
                        question=question,
                        info_summary=final_memory,
                    )
                    self.last_final_answer_source = "normal_generate_fallback"
                    self.last_summary_completeness = "none"
                    self.last_exit_type = "normal_completion"
            else:
                answer = self.generate_answer(
                    question=question,
                    info_summary=final_memory,
                )
                self.last_final_answer_source = "normal_generate_fallback"
                self.last_summary_completeness = "none"
                self.last_exit_type = "normal_completion"

        self._log_progress(
            "[Stage 6/6] Final answer composition DONE",
            start_time=stage_start_time,
            source=self.last_final_answer_source,
            summary_completeness=self.last_summary_completeness,
            exit_type=self.last_exit_type,
        )

        self._log_progress(
            "[DONE] LogicRAG answer_question",
            start_time=total_start_time,
            rounds=round_count,
            dynamic_adaptations=len(stage5_result.get("dynamic_adaptations", []) or []),
            early_stopped=stage5_result.get("early_stopped", False),
            hit_max_rounds=stage5_result.get("hit_max_rounds", False),
        )

        combined_retrieval_history = []
        if warm_up_retrieval_record is not None:
            combined_retrieval_history.append(warm_up_retrieval_record)
        combined_retrieval_history.extend(retrieval_history)

        self.last_dependency_analysis = dependency_analysis_history
        self.last_retrieval_history = combined_retrieval_history

        return answer, last_contexts, round_count
