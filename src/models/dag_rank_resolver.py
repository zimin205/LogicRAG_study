"""
LogicRAG의 DAG rank 단위 retrieval / resolution 모듈.

논문 반영 범위:
- Eq. (1): parent-answer conditioned retrieval
- Eq. (2): subproblem answer generation
- Eq. (3): rolling memory 기반 context pruning
- Eq. (4): same-rank unified query 기반 graph pruning
- Algorithm 1의 rank 순차 처리 및 중간 답 저장
- Algorithm 1 line 14-17의 Dynamic DAG Adaptation
  (run() 메서드에 dag와 max_dynamic_adaptations 인자가 함께 전달된 경우에만 작동.
   기본값으로 호출하면 기존 동작 그대로 유지된다.)

Dynamic DAG Adaptation은 LogicRAG._maybe_add_subproblem()이 담당하고,
resolver는 매 rank 처리 후 해당 메서드를 hook으로 호출한다.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional

from src.utils.utils import get_response_with_retry, fix_json_response


logger = logging.getLogger(__name__)

# Notebook / experiment용 최소 진행 로그 전용 logger.
# 기존 debug logger와 분리해서, Stage 5 rank loop 진행률만 출력한다.
progress_logger = logging.getLogger("logicrag.progress")


def _as_int(value: Any) -> Optional[int]:
    """bool을 제외하고 int 변환 가능한 값만 int로 변환한다."""
    if isinstance(value, bool):
        return None

    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any) -> bool:
    """LLM이 true/false를 문자열로 반환해도 안전하게 bool로 변환한다."""
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False

    return bool(value)


def _clean_text(value: Any) -> str:
    """문자열 값만 정리해서 반환한다."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _normalize_text_key(value: Any) -> str:
    """subproblem text matching용 정규화 key를 만든다."""
    return _clean_text(value).lower()


def build_node_text_by_id(dag_result: Dict[str, Any]) -> Dict[int, str]:
    """
    dag_result에서 node_id -> subproblem text mapping을 만든다.

    dag_result["input_node_ids"]와 dag_result["input_dependencies"]는
    verify_non_cyclicity.py가 같은 순서로 반환한 값이다.
    """
    node_ids = list(dag_result.get("input_node_ids", []) or [])
    dependencies = list(dag_result.get("input_dependencies", []) or [])

    if len(node_ids) != len(dependencies):
        raise ValueError(
            "Invalid dag_result: input_node_ids and input_dependencies length mismatch."
        )

    node_text_by_id: Dict[int, str] = {}

    for idx, raw_node_id in enumerate(node_ids):
        node_id = _as_int(raw_node_id)
        if node_id is None:
            continue

        text = _clean_text(dependencies[idx])
        if not text:
            continue

        node_text_by_id[node_id] = text

    return node_text_by_id


def build_parent_ids_by_node_id(dag_result: Dict[str, Any]) -> Dict[int, List[int]]:
    """
    valid_dependency_edges를 이용해 child node별 parent node 목록을 만든다.

    edge 방향:
        prerequisite_id -> dependent_id

    의미:
        prerequisite_id가 parent node
        dependent_id가 child node
    """
    node_ids = [
        node_id
        for node_id in (
            _as_int(raw_id)
            for raw_id in dag_result.get("input_node_ids", []) or []
        )
        if node_id is not None
    ]

    node_id_set = set(node_ids)

    parent_ids_by_node_id: Dict[int, List[int]] = {
        node_id: []
        for node_id in node_ids
    }

    seen_edges = set()

    for edge in dag_result.get("valid_dependency_edges", []) or []:
        if not isinstance(edge, dict):
            continue

        prerequisite_id = _as_int(edge.get("prerequisite_id"))
        dependent_id = _as_int(edge.get("dependent_id"))

        if prerequisite_id is None or dependent_id is None:
            continue

        if prerequisite_id not in node_id_set or dependent_id not in node_id_set:
            continue

        if prerequisite_id == dependent_id:
            continue

        edge_key = (prerequisite_id, dependent_id)
        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)
        parent_ids_by_node_id[dependent_id].append(prerequisite_id)

    return {
        node_id: sorted(parent_ids)
        for node_id, parent_ids in parent_ids_by_node_id.items()
    }


def build_rank_groups_with_nodes(
    dag_result: Dict[str, Any],
    topological_rank_result: Dict[str, Any],
    sorted_dependencies: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    topological rank 결과를 rank-level resolver에서 쓰기 좋은 형태로 바꾼다.

    반환 형태:
        [
            {
                "rank": 0,
                "nodes": [
                    {"node_id": 0, "subproblem": "..."},
                    {"node_id": 2, "subproblem": "..."}
                ],
                "subproblems": ["...", "..."]
            },
            ...
        ]
    """
    node_text_by_id = build_node_text_by_id(dag_result)

    raw_rank_groups = {}
    if isinstance(topological_rank_result, dict):
        raw_rank_groups = topological_rank_result.get("rank_groups", {}) or {}

    groups: List[Dict[str, Any]] = []

    if isinstance(raw_rank_groups, dict) and raw_rank_groups:
        for raw_rank, raw_node_ids in raw_rank_groups.items():
            rank = _as_int(raw_rank)
            if rank is None:
                continue

            if not isinstance(raw_node_ids, list):
                continue

            nodes: List[Dict[str, Any]] = []
            seen_node_ids = set()

            for raw_node_id in raw_node_ids:
                node_id = _as_int(raw_node_id)
                if node_id is None:
                    continue

                if node_id in seen_node_ids:
                    continue

                subproblem = node_text_by_id.get(node_id, "")
                if not subproblem:
                    continue

                seen_node_ids.add(node_id)
                nodes.append({
                    "node_id": node_id,
                    "subproblem": subproblem,
                })

            nodes = sorted(nodes, key=lambda item: item["node_id"])

            if nodes:
                groups.append({
                    "rank": rank,
                    "nodes": nodes,
                    "subproblems": [node["subproblem"] for node in nodes],
                })

    if not groups:
        sorted_node_ids = [
            node_id
            for node_id in (
                _as_int(raw_id)
                for raw_id in dag_result.get("sorted_node_ids", []) or []
            )
            if node_id is not None
        ]

        if not sorted_node_ids and sorted_dependencies:
            for rank, dependency in enumerate(sorted_dependencies):
                dependency = _clean_text(dependency)
                if not dependency:
                    continue

                groups.append({
                    "rank": rank,
                    "nodes": [
                        {
                            "node_id": rank,
                            "subproblem": dependency,
                        }
                    ],
                    "subproblems": [dependency],
                })

            return groups

        for rank, node_id in enumerate(sorted_node_ids):
            subproblem = node_text_by_id.get(node_id, "")
            if not subproblem:
                continue

            groups.append({
                "rank": rank,
                "nodes": [
                    {
                        "node_id": node_id,
                        "subproblem": subproblem,
                    }
                ],
                "subproblems": [subproblem],
            })

    return sorted(groups, key=lambda item: item["rank"])


def collect_parent_answers_for_nodes(
    nodes: List[Dict[str, Any]],
    parent_ids_by_node_id: Dict[int, List[int]],
    node_text_by_id: Dict[int, str],
    resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
) -> Dict[int, List[Dict[str, Any]]]:
    """
    현재 rank에 있는 각 node에 대해 이미 해결된 parent answer들을 모은다.
    """
    parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]] = {}

    for node in nodes:
        node_id = _as_int(node.get("node_id"))
        if node_id is None:
            continue

        parent_answer_items: List[Dict[str, Any]] = []

        for parent_id in parent_ids_by_node_id.get(node_id, []):
            parent_result = resolved_answers_by_node_id.get(parent_id)
            if not isinstance(parent_result, dict):
                continue

            answer = _clean_text(parent_result.get("answer", ""))
            if not answer:
                continue

            parent_answer_items.append({
                "parent_node_id": parent_id,
                "parent_subproblem": node_text_by_id.get(parent_id, ""),
                "answer": answer,
                "is_answered": _as_bool(parent_result.get("is_answered", False)),
                "evidence_summary": _clean_text(parent_result.get("evidence_summary", "")),
            })

        parent_answers_by_node_id[node_id] = parent_answer_items

    return parent_answers_by_node_id


class ParentConditionedRankResolver:
    """
    LogicRAG Stage 5 담당 클래스.

    담당:
    - 같은 rank의 node들을 묶는다.
    - parent answer를 retrieval query 생성에 반영한다.
    - unified query로 한 번 retrieval한다.
    - retrieved context를 rolling memory로 요약한다.
    - rolling memory로 현재 rank의 node answer를 생성한다.
    - generated answer를 다음 rank용 rolling memory에 반영한다.
    - [확장] 매 rank 처리 후 LogicRAG._maybe_add_subproblem()을 hook으로 호출하여
      Dynamic DAG Adaptation (Algorithm 1 line 14-17)을 수행한다.
    """

    def __init__(self, rag: Any):
        self.rag = rag

    @staticmethod
    def _rank_payload(
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """unified query 생성과 memory 요약 prompt에 넣을 rank payload를 만든다."""
        payload: List[Dict[str, Any]] = []

        for node in nodes:
            node_id = _as_int(node.get("node_id"))
            if node_id is None:
                continue

            payload.append({
                "node_id": node_id,
                "subproblem": _clean_text(node.get("subproblem", "")),
                "resolved_parent_answers": parent_answers_by_node_id.get(node_id, []),
            })

        return payload

    @staticmethod
    def _nodes_payload(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """answer 생성 prompt에 넣을 node 목록을 정리한다."""
        payload: List[Dict[str, Any]] = []

        for node in nodes:
            node_id = _as_int(node.get("node_id"))
            subproblem = _clean_text(node.get("subproblem", ""))

            if node_id is None or not subproblem:
                continue

            payload.append({
                "node_id": node_id,
                "subproblem": subproblem,
            })

        return payload

    @staticmethod
    def _fallback_parent_conditioned_query(
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """LLM query merge가 실패했을 때 쓰는 deterministic fallback query."""
        payload = ParentConditionedRankResolver._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        lines = [
            f"Original question: {question}",
            f"Topological rank: {rank}",
            "Retrieve evidence for the following same-rank subproblems.",
        ]

        for item in payload:
            lines.append(f"- Node {item['node_id']}: {item['subproblem']}")

            parent_answers = item.get("resolved_parent_answers", [])
            if parent_answers:
                lines.append("  Resolved parent answers for retrieval conditioning:")
                for parent in parent_answers:
                    lines.append(
                        f"  - Parent node {parent['parent_node_id']}: "
                        f"{parent['answer']} "
                        f"(parent subproblem: {parent['parent_subproblem']})"
                    )

        return "\n".join(lines)

    def build_parent_conditioned_unified_query(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
    ) -> str:
        """논문 Eq. (1)과 Eq. (4)를 함께 구현한다."""
        nodes = [
            {
                "node_id": _as_int(node.get("node_id")),
                "subproblem": _clean_text(node.get("subproblem", "")),
            }
            for node in nodes
        ]

        nodes = [
            node
            for node in nodes
            if node["node_id"] is not None and node["subproblem"]
        ]

        if not nodes:
            return question

        has_parent_answers = any(
            parent_answers_by_node_id.get(node["node_id"])
            for node in nodes
        )

        if len(nodes) == 1 and not has_parent_answers:
            return nodes[0]["subproblem"]

        fallback_query = self._fallback_parent_conditioned_query(
            question=question,
            rank=rank,
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        rank_payload = self._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        prompt = f"""
You are constructing ONE retrieval query for a topological rank in LogicRAG.

Original question Q:
{question}

Topological rank r:
{rank}

Same-rank subproblems S(r) with resolved parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Task:
Construct a unified retrieval query q(r) for S(r).

Rules:
- Merge same-rank subproblems into one query.
- Use resolved parent answers as concrete anchors when they exist.
- Preserve every entity, relation, date constraint, comparison target, and requested attribute.
- Use parent answers only to make the retrieval query concrete.
- Do not answer the subproblems.
- Do not introduce entities not present in the subproblems or parent answers.
- Prefer a concise factoid-style retrieval query.
- Return ONLY a JSON object.

Output schema:
{{
  "unified_query": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if isinstance(parsed, dict):
                unified_query = parsed.get("unified_query", "")
                if isinstance(unified_query, str) and unified_query.strip():
                    return unified_query.strip()

        except Exception as e:
            logger.error("Error generating parent-conditioned unified query: %s", e)

        return fallback_query

    def summarize_rank_context_to_memory(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        parent_answers_by_node_id: Dict[int, List[Dict[str, Any]]],
        unified_query: str,
        contexts: List[str],
        previous_memory: str = "",
    ) -> str:
        """논문 Eq. (3)과 Algorithm 1의 rolling memory update를 구현한다."""
        context_text = "\n\n".join(contexts or [])

        rank_payload = self._rank_payload(
            nodes=nodes,
            parent_answers_by_node_id=parent_answers_by_node_id,
        )

        prompt = f"""
You are updating LogicRAG rolling memory.

Original question Q:
{question}

Previous rolling memory Mem(r-1):
{previous_memory}

Topological rank r:
{rank}

Same-rank subproblems S(r) with retrieval-conditioning parent answers:
{json.dumps(rank_payload, ensure_ascii=False, indent=2)}

Unified query q(r):
{unified_query}

Retrieved documents C(r):
{context_text}

Task:
Produce Mem(r) by summarizing Mem(r-1) and C(r) with respect to Q.

Rules:
- Keep only salient facts useful for resolving the current rank, later dependent subproblems, or the final answer.
- Preserve exact entity names, dates, numbers, relations, and comparison targets.
- Remove irrelevant, redundant, or noisy context.
- Do not answer the final original question.
- Do not invent unsupported facts.
- Do not copy full retrieved passages.
- Return ONLY a JSON object.

Output schema:
{{
  "memory": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if isinstance(parsed, dict):
                memory = parsed.get("memory", "")
                if isinstance(memory, str) and memory.strip():
                    return memory.strip()

        except Exception as e:
            logger.error("Error summarizing rank context to rolling memory: %s", e)

        fallback_parts = []

        if previous_memory:
            fallback_parts.append(previous_memory)

        fallback_parts.append(f"Rank {rank} unified query: {unified_query}")
        fallback_parts.append(f"Rank {rank} retrieved context:\n{context_text}")

        return "\n\n".join(fallback_parts).strip()

    def resolve_rank_with_memory(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        unified_query: str,
        memory: str,
    ) -> Dict[str, Any]:
        """논문 Algorithm 1의 subproblem resolution 단계를 구현한다."""
        nodes = [
            {
                "node_id": _as_int(node.get("node_id")),
                "subproblem": _clean_text(node.get("subproblem", "")),
            }
            for node in nodes
        ]

        nodes = [
            node
            for node in nodes
            if node["node_id"] is not None and node["subproblem"]
        ]

        fallback_answers = [
            {
                "node_id": node["node_id"],
                "subproblem": node["subproblem"],
                "answer": "",
                "is_answered": False,
                "evidence_summary": "",
                "missing_info": "No parsed answer was produced.",
            }
            for node in nodes
        ]

        if not nodes:
            return {
                "node_answers": [],
                "rank_summary": "",
            }

        nodes_payload = self._nodes_payload(nodes)

        prompt = f"""
You are resolving same-rank subproblems in LogicRAG.

Original question Q:
{question}

Topological rank r:
{rank}

Same-rank subproblems S(r):
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Unified retrieval query q(r):
{unified_query}

Current rolling memory Mem(r):
{memory}

Task:
For each node, answer only that node's subproblem using Mem(r).

Rules:
- Do not answer the final original question.
- Return one item for every node in the same order.
- Use node_id exactly as provided.
- Use only Mem(r).
- Do not use outside knowledge.
- Do not invent unsupported facts.
- If Mem(r) is insufficient, set is_answered to false and explain missing_info.
- Keep each answer concise.
- Return ONLY a JSON object.

Output schema:
{{
  "node_answers": [
    {{
      "node_id": integer,
      "subproblem": string,
      "answer": string,
      "is_answered": boolean,
      "evidence_summary": string,
      "missing_info": string
    }}
  ],
  "rank_summary": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if not isinstance(parsed, dict):
                raise ValueError("Rank resolution response is not a dict.")

            raw_answers = parsed.get("node_answers", [])
            if not isinstance(raw_answers, list):
                raw_answers = []

            node_ids = {node["node_id"] for node in nodes}
            text_to_node_id = {
                _normalize_text_key(node["subproblem"]): node["node_id"]
                for node in nodes
            }

            answer_by_node_id: Dict[int, Dict[str, Any]] = {}

            for raw_answer in raw_answers:
                if not isinstance(raw_answer, dict):
                    continue

                node_id = _as_int(raw_answer.get("node_id"))

                if node_id is None:
                    node_id = text_to_node_id.get(
                        _normalize_text_key(raw_answer.get("subproblem", ""))
                    )

                if node_id not in node_ids:
                    continue

                original_subproblem = next(
                    node["subproblem"]
                    for node in nodes
                    if node["node_id"] == node_id
                )

                answer_by_node_id[node_id] = {
                    "node_id": node_id,
                    "subproblem": original_subproblem,
                    "answer": _clean_text(raw_answer.get("answer", "")),
                    "is_answered": _as_bool(raw_answer.get("is_answered", False)),
                    "evidence_summary": _clean_text(raw_answer.get("evidence_summary", "")),
                    "missing_info": _clean_text(raw_answer.get("missing_info", "")),
                }

            normalized_answers: List[Dict[str, Any]] = []

            for node in nodes:
                node_id = node["node_id"]
                answer = answer_by_node_id.get(node_id)

                if answer is None:
                    answer = {
                        "node_id": node_id,
                        "subproblem": node["subproblem"],
                        "answer": "",
                        "is_answered": False,
                        "evidence_summary": "",
                        "missing_info": "No answer mapped to this node.",
                    }

                normalized_answers.append(answer)

            rank_summary = parsed.get("rank_summary", "")
            if not isinstance(rank_summary, str):
                rank_summary = ""

            return {
                "node_answers": normalized_answers,
                "rank_summary": rank_summary.strip(),
            }

        except Exception as e:
            logger.error("Error resolving rank with rolling memory: %s", e)

            return {
                "node_answers": fallback_answers,
                "rank_summary": "",
            }

    def distill_rank_result_to_memory(
        self,
        question: str,
        rank: int,
        nodes: List[Dict[str, Any]],
        unified_query: str,
        contexts: List[str],
        memory_for_resolution: str,
        rank_result: Dict[str, Any],
    ) -> str:
        """Framework 본문의 context pruning 설명을 반영한다."""
        context_text = "\n\n".join(contexts or [])
        nodes_payload = self._nodes_payload(nodes)
        rank_result_text = json.dumps(rank_result, ensure_ascii=False, indent=2)

        prompt = f"""
You are updating LogicRAG rolling memory after resolving a topological rank.

Original question Q:
{question}

Topological rank r:
{rank}

Same-rank subproblems S(r):
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Unified query q(r):
{unified_query}

Retrieved documents C(r):
{context_text}

Memory used for resolution Mem(r):
{memory_for_resolution}

Resolved intermediate answers for this rank:
{rank_result_text}

Task:
Distill the retrieved context and the generated intermediate answers into the rolling memory for subsequent ranks.

Rules:
- Preserve only salient facts needed for later dependent subproblems or final answer composition.
- Preserve exact entity names, dates, numbers, relations, and comparison targets.
- Preserve generated intermediate answers when they are supported by the memory/context.
- Remove irrelevant, redundant, or noisy details.
- Do not answer the final original question.
- Do not invent unsupported facts.
- Return ONLY a JSON object.

Output schema:
{{
  "memory": string
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if isinstance(parsed, dict):
                memory = parsed.get("memory", "")
                if isinstance(memory, str) and memory.strip():
                    return memory.strip()

        except Exception as e:
            logger.error("Error distilling rank result to rolling memory: %s", e)

        fallback_parts = []

        if memory_for_resolution:
            fallback_parts.append(memory_for_resolution)

        fallback_parts.append(f"Rank {rank} result: {rank_result_text}")

        return "\n\n".join(fallback_parts).strip()

    @staticmethod
    def build_final_subanswer_summary(
        dag_result: Dict[str, Any],
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
    ) -> str:
        """최종 answer composition에 넣을 node별 중간 답 요약을 만든다."""
        sorted_node_ids = [
            node_id
            for node_id in (
                _as_int(raw_id)
                for raw_id in dag_result.get("sorted_node_ids", []) or []
            )
            if node_id is not None
        ]

        if not sorted_node_ids:
            sorted_node_ids = sorted(resolved_answers_by_node_id.keys())

        items: List[Dict[str, Any]] = []

        for node_id in sorted_node_ids:
            answer_item = resolved_answers_by_node_id.get(node_id)
            if not isinstance(answer_item, dict):
                continue

            items.append({
                "node_id": node_id,
                "subproblem": answer_item.get("subproblem", ""),
                "answer": answer_item.get("answer", ""),
                "is_answered": _as_bool(answer_item.get("is_answered", False)),
                "evidence_summary": answer_item.get("evidence_summary", ""),
                "missing_info": answer_item.get("missing_info", ""),
            })

        return json.dumps(items, ensure_ascii=False, indent=2)

    @staticmethod
    def build_final_subanswer_summary_from_processed_groups(
        processed_rank_groups: List[Dict[str, Any]],
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
    ) -> str:
        """
        최종 answer composition에 넣을 node별 중간 답 요약을 만든다.

        Dynamic DAG Adaptation으로 새로 append된 node는 초기 dag_result["sorted_node_ids"]에
        존재하지 않는다. 따라서 최종 Compose 단계에서는 실제로 처리된 rank group 순서
        (processed_rank_groups)를 기준으로 subanswer summary를 만들어야 한다.
        """
        ordered_node_ids: List[int] = []
        seen_node_ids = set()

        for group in processed_rank_groups or []:
            for node in group.get("nodes", []) or []:
                if not isinstance(node, dict):
                    continue

                node_id = _as_int(node.get("node_id"))
                if node_id is None or node_id in seen_node_ids:
                    continue

                seen_node_ids.add(node_id)
                ordered_node_ids.append(node_id)

        if not ordered_node_ids:
            ordered_node_ids = sorted(resolved_answers_by_node_id.keys())

        items: List[Dict[str, Any]] = []

        for node_id in ordered_node_ids:
            answer_item = resolved_answers_by_node_id.get(node_id)
            if not isinstance(answer_item, dict):
                continue

            items.append({
                "node_id": node_id,
                "subproblem": answer_item.get("subproblem", ""),
                "answer": answer_item.get("answer", ""),
                "is_answered": _as_bool(answer_item.get("is_answered", False)),
                "evidence_summary": answer_item.get("evidence_summary", ""),
                "missing_info": answer_item.get("missing_info", ""),
            })

        return json.dumps(items, ensure_ascii=False, indent=2)

    def judge_can_answer_now(
        self,
        question: str,
        memory: str,
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]],
        processed_rank_groups: List[Dict[str, Any]],
        remaining_rank_groups: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Determine whether rank processing can stop now.

        This is the rank-level counterpart of official LogicRAG's can_answer
        gate. It deliberately uses only a boolean can_answer, not a confidence
        label, so early stopping has a single auditable decision criterion.
        """
        resolved_payload = [
            answer
            for _, answer in sorted(
                resolved_answers_by_node_id.items(),
                key=lambda item: item[0],
            )
            if isinstance(answer, dict)
        ]

        remaining_payload = []
        for group in remaining_rank_groups or []:
            if not isinstance(group, dict):
                continue

            remaining_payload.append({
                "rank": group.get("rank"),
                "nodes": group.get("nodes", []),
            })

        prompt = f"""You are deciding whether LogicRAG can stop early.

Original question:
{question}

Current rolling memory:
{memory}

Resolved subproblem answers so far:
{json.dumps(resolved_payload, ensure_ascii=False, indent=2)}

Processed rank groups:
{json.dumps(processed_rank_groups, ensure_ascii=False, indent=2)}

Remaining unprocessed rank groups:
{json.dumps(remaining_payload, ensure_ascii=False, indent=2)}

Task:
Decide whether the original question can now be answered completely using ONLY the current rolling memory and resolved subproblem answers.

Return can_answer=true ONLY if:
- The final answer to the original question can be produced now.
- No additional retrieval is needed.
- No required factual dependency is missing.
- Remaining unprocessed nodes are not needed to determine the final answer, or only require trivial composition already possible from resolved facts.

If any remaining node may change or complete the final answer, return can_answer=false.

Return ONLY a JSON object with this schema:
{{
  "can_answer": boolean,
  "current_understanding": string,
  "missing_info": string,
  "blocking_remaining_node_ids": [list of integers],
  "final_answer": string or null
}}
"""

        try:
            response = get_response_with_retry(prompt)
            response = response.strip().replace("```json", "").replace("```", "")
            parsed = fix_json_response(response)

            if not isinstance(parsed, dict):
                raise ValueError("early-stop judgement response is not a JSON object")

            parsed["can_answer"] = _as_bool(parsed.get("can_answer", False))

            if not isinstance(parsed.get("blocking_remaining_node_ids", []), list):
                parsed["blocking_remaining_node_ids"] = []

            parsed.setdefault("current_understanding", "")
            parsed.setdefault("missing_info", "")
            parsed.setdefault("final_answer", None)

            return parsed

        except Exception as e:
            logger.error("Error in early-stop judgement: %s", e)
            return {
                "can_answer": False,
                "current_understanding": "",
                "missing_info": f"Early-stop judgement failed: {e}",
                "blocking_remaining_node_ids": [],
                "final_answer": None,
            }

    def run(
        self,
        question: str,
        dag_result: Dict[str, Any],
        topological_rank_result: Dict[str, Any],
        sorted_dependencies: Optional[List[str]] = None,
        initial_memory: str = "",
        retrieved_chunks_set: Optional[set] = None,
        max_rounds: Optional[int] = None,
        # [추가] Dynamic DAG Adaptation을 위한 옵셔널 인자
        # dag와 max_dynamic_adaptations가 함께 제공된 경우에만 dynamic adaptation이 작동한다.
        # 기본값으로 호출하면 기존 동작 그대로 유지된다 (하위 호환성).
        dag: Optional[Any] = None,
        max_dynamic_adaptations: int = 0,
        enable_early_stop: bool = False,
    ) -> Dict[str, Any]:
        """
        Parent-answer conditioned rank-level LogicRAG resolution.

        매 rank 처리 후 Dynamic DAG Adaptation hook 호출:
        - dag와 max_dynamic_adaptations가 함께 제공된 경우에만 발동.
        - LogicRAG._maybe_add_subproblem()을 호출하여 새 sub 필요 여부 판정.
        - 새 sub가 추가되면 rank_groups_to_process에 append하여 루프가 자동 연장된다.
        - max_dynamic_adaptations 안전장치로 무한 추가 방지.

        Args:
            question: 원래 질문 Q.
            dag_result: verify_non_cyclicity의 검증 결과 dict.
            topological_rank_result: dag_topological_rank의 결과 dict.
            sorted_dependencies: fallback용 정렬된 의존 리스트.
            initial_memory: Mem(0). 논문 baseline에서는 ∅(빈 문자열).
            retrieved_chunks_set: filter_repeats용 retrieved chunk 집합.
            max_rounds: rank 처리 최대 횟수. None이면 모든 rank 처리.
            dag: QueryLogicDAG 객체. dynamic adaptation 시 mutate된다.
            max_dynamic_adaptations: dynamic adaptation 최대 발동 횟수.
            enable_early_stop: 각 rank 처리 후 can_answer 조기종료 gate를 켤지 여부.

        Returns:
            stage5 결과 dict. 다음 키 포함:
            - rank_groups, processed_rank_groups, resolved_answers_by_node_id,
              retrieval_history, last_contexts, round_count, final_memory,
              final_subanswer_summary, dynamic_adaptations
        """
        if not dag_result.get("is_dag", False):
            raise ValueError("Parent-conditioned rank resolution requires a valid DAG.")

        node_text_by_id = build_node_text_by_id(dag_result)
        parent_ids_by_node_id = build_parent_ids_by_node_id(dag_result)

        rank_groups = build_rank_groups_with_nodes(
            dag_result=dag_result,
            topological_rank_result=topological_rank_result,
            sorted_dependencies=sorted_dependencies,
        )

        if max_rounds is not None:
            max_rounds = max(0, int(max_rounds))

        rank_groups_to_process = list(rank_groups)
        scheduled_rounds = (
            len(rank_groups_to_process)
            if max_rounds is None
            else min(len(rank_groups_to_process), max_rounds)
        )

        if progress_logger.isEnabledFor(logging.INFO):
            progress_logger.info(
                "[Stage 5/6] Rank groups prepared | rank_groups=%d | scheduled_rounds=%d | max_rounds=%s | early_stop=%s",
                len(rank_groups),
                scheduled_rounds,
                max_rounds,
                enable_early_stop,
            )

        memory = initial_memory or ""
        resolved_answers_by_node_id: Dict[int, Dict[str, Any]] = {}
        retrieval_history: List[Dict[str, Any]] = []
        last_contexts: List[str] = []

        # [추가] Dynamic DAG Adaptation 상태 변수
        dynamic_adaptations_log: List[Dict[str, Any]] = []
        adaptation_count = 0

        processed_rank_groups: List[Dict[str, Any]] = []
        early_stopped = False
        early_stop_result: Optional[Dict[str, Any]] = None

        # [변경] for-enumerate 대신 while-index 패턴.
        # rank_groups_to_process에 매 iteration 끝에 append할 수 있으므로
        # 명시적으로 list 길이를 매번 확인한다.
        idx = 0
        while (
            idx < len(rank_groups_to_process)
            and (max_rounds is None or len(retrieval_history) < max_rounds)
        ):
            rank_group = rank_groups_to_process[idx]
            round_idx = len(retrieval_history) + 1

            rank = int(rank_group["rank"])
            nodes = rank_group.get("nodes", []) or []
            subproblems = rank_group.get("subproblems", []) or [
                node.get("subproblem", "")
                for node in nodes
            ]

            round_start_time = time.perf_counter()

            if progress_logger.isEnabledFor(logging.INFO):
                progress_logger.info(
                    "[Stage 5/6][Round %d/%d] START | rank=%s | nodes=%d",
                    round_idx,
                    len(rank_groups_to_process),
                    rank,
                    len(nodes),
                )

            memory_before_rank = memory

            # 1. 부모 node 답을 수집한다.
            parent_answers_by_node_id = collect_parent_answers_for_nodes(
                nodes=nodes,
                parent_ids_by_node_id=parent_ids_by_node_id,
                node_text_by_id=node_text_by_id,
                resolved_answers_by_node_id=resolved_answers_by_node_id,
            )

            # 2. 같은 rank의 subproblem을 하나의 unified query로 묶는다.
            unified_query = self.build_parent_conditioned_unified_query(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
            )

            # 3. unified query로 retrieval을 한 번 수행한다.
            contexts = self.rag._retrieve_for_query(
                unified_query,
                retrieved_chunks_set=retrieved_chunks_set,
            )
            last_contexts = contexts

            # 4. retrieved context를 이전 memory와 합쳐 현재 rank용 memory로 요약한다.
            memory_for_resolution = self.summarize_rank_context_to_memory(
                question=question,
                rank=rank,
                nodes=nodes,
                parent_answers_by_node_id=parent_answers_by_node_id,
                unified_query=unified_query,
                contexts=contexts,
                previous_memory=memory_before_rank,
            )

            # 5. 현재 rank의 각 subproblem answer는 rolling memory로 생성한다.
            rank_result = self.resolve_rank_with_memory(
                question=question,
                rank=rank,
                nodes=nodes,
                unified_query=unified_query,
                memory=memory_for_resolution,
            )

            # 6. node_id 기준으로 중간 답을 저장한다.
            for node_answer in rank_result.get("node_answers", []) or []:
                node_id = _as_int(node_answer.get("node_id"))
                if node_id is None:
                    continue

                resolved_answers_by_node_id[node_id] = node_answer

            # 7. retrieved context와 generated answer를 다음 rank용 rolling memory에 반영한다.
            memory_after_rank = self.distill_rank_result_to_memory(
                question=question,
                rank=rank,
                nodes=nodes,
                unified_query=unified_query,
                contexts=contexts,
                memory_for_resolution=memory_for_resolution,
                rank_result=rank_result,
            )

            memory = memory_after_rank

            retrieval_history.append({
                "round": round_idx,
                "rank": rank,
                "nodes": nodes,
                "subproblems": subproblems,
                "parent_answers_by_node_id": parent_answers_by_node_id,
                "unified_query": unified_query,
                "contexts": contexts,
                "memory_before_rank": memory_before_rank,
                "memory_for_resolution": memory_for_resolution,
                "memory_after_rank": memory_after_rank,
                "rank_result": rank_result,
            })

            processed_rank_groups.append(rank_group)

            # ─── 8. [추가] can_answer 기반 early stopping hook ───
            # 공식 LogicRAG의 per-round can_answer early return을
            # DAG-rank 구조에서는 rank 처리 직후, dynamic adaptation 전에 수행한다.
            if enable_early_stop:
                remaining_rank_groups = rank_groups_to_process[idx + 1:]

                early_judgement = self.judge_can_answer_now(
                    question=question,
                    memory=memory,
                    resolved_answers_by_node_id=resolved_answers_by_node_id,
                    processed_rank_groups=processed_rank_groups,
                    remaining_rank_groups=remaining_rank_groups,
                )

                retrieval_history[-1]["early_stop_judgement"] = early_judgement

                if _as_bool(early_judgement.get("can_answer", False)):
                    early_stopped = True
                    early_stop_result = {
                        "round": round_idx,
                        "rank": rank,
                        "judgement": early_judgement,
                    }

                    if progress_logger.isEnabledFor(logging.INFO):
                        progress_logger.info(
                            "[Stage 5/6][Round %d] EARLY STOP | rank=%s",
                            round_idx,
                            rank,
                        )

                    idx += 1
                    break

            # ─── 9. [추가] Dynamic DAG Adaptation hook ───
            # 논문 Algorithm 1 line 14-17 구현.
            # dag와 max_dynamic_adaptations가 함께 제공된 경우에만 작동.
            # LogicRAG._maybe_add_subproblem()을 호출하여 새 sub 필요 여부 판정.
            # 새 sub가 추가되면 rank_groups_to_process에 append하여 다음 iteration에서 처리됨.
            has_round_budget_after_this = (
                max_rounds is None
                or len(retrieval_history) < max_rounds
            )

            if (
                has_round_budget_after_this
                and dag is not None
                and max_dynamic_adaptations > 0
                and adaptation_count < max_dynamic_adaptations
            ):
                # 부모 rank 조회용으로 현재까지의 max rank 계산
                current_max_rank = max(
                    (int(g["rank"]) for g in rank_groups_to_process),
                    default=rank,
                )

                # node_id -> answer string 형태로 변환
                # (LogicRAG._maybe_add_subproblem이 요구하는 sub_answers 형식)
                sub_answers_for_hook: Dict[int, str] = {}
                for node_id, answer_dict in resolved_answers_by_node_id.items():
                    if not isinstance(answer_dict, dict):
                        continue
                    answer_text = _clean_text(answer_dict.get("answer", ""))
                    if answer_text:
                        sub_answers_for_hook[node_id] = answer_text

                unresolved_answers_for_hook = [
                    answer
                    for answer in rank_result.get("node_answers", []) or []
                    if isinstance(answer, dict)
                    and (
                        not _as_bool(answer.get("is_answered", False))
                        or bool(_clean_text(answer.get("missing_info", "")))
                    )
                ]

                try:
                    new_sub_info = self.rag._maybe_add_subproblem(
                        question=question,
                        info_summary=memory,
                        dag=dag,
                        sub_answers=sub_answers_for_hook,
                        current_max_rank=current_max_rank,
                        unresolved_answers=unresolved_answers_for_hook,
                        current_rank=rank,
                        current_nodes=nodes,
                        rank_result=rank_result,
                    )
                except Exception as e:
                    logger.error(
                        "Error during dynamic adaptation hook: %s", e
                    )
                    new_sub_info = None

                if new_sub_info is not None:
                    new_id = new_sub_info["new_subproblem_id"]
                    new_rank = new_sub_info["new_rank"]
                    new_text = new_sub_info["new_subproblem_text"]

                    # 새 rank group을 list에 append하여 루프가 자동 연장되게 함
                    new_rank_group = {
                        "rank": new_rank,
                        "nodes": [{
                            "node_id": new_id,
                            "subproblem": new_text,
                        }],
                        "subproblems": [new_text],
                    }
                    rank_groups_to_process.append(new_rank_group)

                    # 다음 rank 처리 시 부모 답 조회를 위해 mapping 업데이트
                    node_text_by_id[new_id] = new_text
                    parent_ids_by_node_id[new_id] = list(new_sub_info.get("depends_on", []))

                    adaptation_count += 1
                    dynamic_adaptations_log.append({
                        "after_round": round_idx,
                        "after_rank": rank,
                        "adaptation_index": adaptation_count,
                        "added_subproblem": new_sub_info,
                    })

            if progress_logger.isEnabledFor(logging.INFO):
                progress_logger.info(
                    "[Stage 5/6][Round %d/%d] DONE | rank=%s | elapsed=%.2fs | contexts=%d | memory_chars=%d | dynamic_adaptations=%d",
                    round_idx,
                    len(rank_groups_to_process),
                    rank,
                    time.perf_counter() - round_start_time,
                    len(contexts or []),
                    len(memory or ""),
                    adaptation_count,
                )

            idx += 1

        final_subanswer_summary = self.build_final_subanswer_summary_from_processed_groups(
            processed_rank_groups=processed_rank_groups,
            resolved_answers_by_node_id=resolved_answers_by_node_id,
        )

        unprocessed_rank_groups = rank_groups_to_process[idx:]
        hit_max_rounds = (
            max_rounds is not None
            and len(retrieval_history) >= max_rounds
            and bool(unprocessed_rank_groups)
            and not early_stopped
        )

        if progress_logger.isEnabledFor(logging.INFO):
            progress_logger.info(
                "[Stage 5/6] Resolver completed | rounds=%d | dynamic_adaptations=%d | final_memory_chars=%d",
                len(retrieval_history),
                len(dynamic_adaptations_log),
                len(memory or ""),
            )

        return {
            "rank_groups": rank_groups,
            "processed_rank_groups": processed_rank_groups,
            "unprocessed_rank_groups": unprocessed_rank_groups,
            "resolved_answers_by_node_id": resolved_answers_by_node_id,
            "retrieval_history": retrieval_history,
            "last_contexts": last_contexts,
            "round_count": len(retrieval_history),
            "final_memory": memory,
            "final_subanswer_summary": final_subanswer_summary,
            # [추가] Dynamic DAG Adaptation 발동 이력
            "dynamic_adaptations": dynamic_adaptations_log,
            "early_stopped": early_stopped,
            "early_stop_result": early_stop_result,
            "hit_max_rounds": hit_max_rounds,
            "max_rounds": max_rounds,
        }