import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from openai import APIError

from src.utils.utils import get_response_with_retry, fix_json_response


logger = logging.getLogger(__name__)


# ============================================================
# 논문 기반 고정 문자열들
# ------------------------------------------------------------
# 아래 값들은 알고리즘 계산값이라기보다는 "기록용 라벨"이다.
# 즉, 나중에 로그를 봤을 때 이 DAG, node, edge가
# 논문 흐름의 어떤 단계에서 만들어졌는지 추적하기 위해 사용한다.
# ============================================================

# 이 그래프가 논문에서 말하는 Query Logic Dependency Graph임을 표시한다.
GRAPH_TYPE_QUERY_LOGIC_DEPENDENCY_GRAPH = "query_logic_dependency_graph"

# node가 처음 query decomposition 단계에서 나온 subproblem임을 표시한다.
NODE_SOURCE_INITIAL_DECOMPOSITION = "initial_decomposition"

# node가 기존 코드의 dependencies 리스트에서 임시로 만들어졌음을 표시한다.
# 논문 정식 흐름은 아니고, 현재 repo와 연결하기 위한 기존 코드 호환용이다.
NODE_SOURCE_EXISTING_DEPENDENCY_TEXTS = "existing_dependency_texts"

# node를 만든 단계가 query decomposition임을 표시한다.
NODE_CREATED_BY_QUERY_DECOMPOSITION = "query_decomposition"

# node를 만든 단계가 기존 warm_up_analysis 결과임을 표시한다.
# 이것도 논문 정식 흐름이 아니라 기존 코드 호환용이다.
NODE_CREATED_BY_WARM_UP_ANALYSIS = "warm_up_analysis"

# edge가 논문에서 말하는 logical precedence 관계임을 표시한다.
# logical precedence = "A를 먼저 풀어야 B를 풀 수 있다"는 논리적 선후관계.
EDGE_RELATION_LOGICAL_PRECEDENCE = "logical_precedence"

# edge가 LLM dependency modeling 단계에서 추론되었음을 표시한다.
EDGE_SOURCE_LLM_DEPENDENCY_MODELING = "llm_dependency_modeling"

# edge가 초기 graph construction 단계에서 만들어졌음을 표시한다.
EDGE_CREATED_BY_INITIAL_GRAPH_CONSTRUCTION = "initial_graph_construction"


@dataclass
class SubproblemNode:
    """
    Query Logic Dependency Graph의 node.

    논문 근거:
        입력 query Q는 subproblem 집합 P={p1, p2, ..., pn}으로 분해된다.
        각 subproblem p_i는 DAG G=(V,E)의 node v_i에 대응된다.

    필드 설명:
        id:
            node 번호.
            보통 subproblem의 id 또는 리스트 순서를 사용한다.

        text:
            실제 subproblem 문장.

        metadata:
            이 node가 어디서 만들어졌는지 기록하는 보조 정보.
            예:
                - 처음 query decomposition에서 나왔는지
                - 기존 코드의 dependencies에서 임시로 만들어졌는지
                - 원래 subproblem 리스트에서 몇 번째였는지
    """
    id: int
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyEdge:
    """
    Query Logic Dependency Graph의 directed edge.

    방향:
        prerequisite_id -> dependent_id

    의미:
        prerequisite_id에 해당하는 subproblem을 먼저 풀어야
        dependent_id에 해당하는 subproblem을 풀 수 있다.

    예:
        0번: "프랑스의 수도 찾기"
        1번: "그 수도의 시장 찾기"

        0번을 먼저 풀어야 1번을 풀 수 있으므로:
            0 -> 1

    필드 설명:
        prerequisite_id:
            먼저 풀어야 하는 node id.

        dependent_id:
            prerequisite 결과에 의존하는 node id.

        reason:
            왜 이 edge가 필요한지에 대한 설명.
            논문 알고리즘의 필수 수식 요소는 아니지만,
            LLM이 만든 edge를 검토하기 위한 해석/디버깅 정보다.

        metadata:
            이 edge가 어떤 관계인지, 어떤 단계에서 만들어졌는지 기록한다.
            기본적으로 relation_type은 logical_precedence로 둔다.
    """
    prerequisite_id: int
    dependent_id: int
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryLogicDAG:
    """
    Query Logic Dependency Graph G=(V,E).

    논문 근거:
        - Q는 입력 query다.
        - P={p1, p2, ..., pn}은 Q에서 분해된 subproblem 집합이다.
        - V={v1, v2, ..., vn}은 subproblem을 나타내는 node 집합이다.
        - E는 subproblem 사이의 logical dependency를 나타내는 directed edge 집합이다.
        - G=(V,E)는 directed acyclic graph, 즉 DAG로 사용된다.

    필드 설명:
        V:
            node 집합.
            key는 node id, value는 SubproblemNode.

        E:
            edge 집합.
            각 edge는 prerequisite_id -> dependent_id 방향을 가진다.

        metadata:
            DAG 전체에 대한 정보.
            예:
                - 이 DAG가 어떤 original query Q에서 만들어졌는지
                - 이 그래프 종류가 Query Logic Dependency Graph인지
                - acyclicity 검증이 되었는지

        parents:
            각 node의 부모 node id 집합.
            부모는 "나보다 먼저 풀려야 하는 node"를 의미한다.

        children:
            각 node의 자식 node id 집합.
            자식은 "나를 먼저 풀어야 풀 수 있는 node"를 의미한다.

        topological_order:
            나중에 topological-sort 모듈이 채울 실행 순서.
            이 Builder는 cycle 검사를 직접 하지 않는다.

        ranks:
            나중에 graph pruning 또는 rank 계산 모듈이 채울 rank 정보.
    """
    V: Dict[int, SubproblemNode]
    E: List[DependencyEdge]
    metadata: Dict[str, Any] = field(default_factory=dict)

    parents: Dict[int, Set[int]] = field(default_factory=dict)
    children: Dict[int, Set[int]] = field(default_factory=dict)

    # topological-sort / rank 모듈에서 나중에 채우는 값이다.
    topological_order: List[int] = field(default_factory=list)
    ranks: Dict[int, int] = field(default_factory=dict)

    def rebuild_indexes(self) -> None:
        """
        edge 목록 E를 기준으로 parents/children lookup table을 다시 만든다.

        왜 필요한가:
            E만 있으면 "이 node의 부모가 누구인지" 매번 edge 전체를 뒤져야 한다.
            parents/children을 만들어두면 이후 retrieval planning이나
            topological sort 단계에서 빠르게 조회할 수 있다.
        """
        self.parents = {node_id: set() for node_id in self.V}
        self.children = {node_id: set() for node_id in self.V}

        for edge in self.E:
            self.children[edge.prerequisite_id].add(edge.dependent_id)
            self.parents[edge.dependent_id].add(edge.prerequisite_id)

    def get_parents(self, node_id: int) -> List[SubproblemNode]:
        """
        특정 node의 부모 node들을 반환한다.

        부모 node 의미:
            현재 node를 풀기 전에 먼저 풀려야 하는 subproblem들.
        """
        return [
            self.V[parent_id]
            for parent_id in self.parents.get(node_id, set())
        ]

    def get_children(self, node_id: int) -> List[SubproblemNode]:
        """
        특정 node의 자식 node들을 반환한다.

        자식 node 의미:
            현재 node의 결과에 의존하는 subproblem들.
        """
        return [
            self.V[child_id]
            for child_id in self.children.get(node_id, set())
        ]

    def get_root_nodes(self) -> List[SubproblemNode]:
        """
        부모가 없는 node들을 반환한다.

        의미:
            다른 subproblem의 결과 없이 바로 풀 수 있는 시작 subproblem들.
        """
        return [
            self.V[node_id]
            for node_id in self.V
            if not self.parents.get(node_id)
        ]

    def get_leaf_nodes(self) -> List[SubproblemNode]:
        """
        자식이 없는 node들을 반환한다.

        의미:
            다른 subproblem이 더 이상 의존하지 않는 마지막 단계의 subproblem들.
            보통 최종 비교, 최종 종합, 최종 판단에 가까운 node일 수 있다.
        """
        return [
            self.V[node_id]
            for node_id in self.V
            if not self.children.get(node_id)
        ]

    def add_node(self, node: SubproblemNode) -> None:
        """
        DAG에 새 subproblem node를 추가한다.

        논문 근거:
            LogicRAG는 inference 중 retrieval context가 부족하면
            새로운 subproblem을 추가하여 DAG를 동적으로 확장할 수 있다.

        주의:
            이 함수는 node만 추가한다.
            새 node와 기존 node 사이의 dependency edge는 add_edge()로 따로 추가해야 한다.
        """
        if node.id in self.V:
            raise ValueError(f"Node id already exists: {node.id}")

        self.V[node.id] = node
        self.parents[node.id] = set()
        self.children[node.id] = set()

    def add_edge(self, edge: DependencyEdge) -> None:
        """
        DAG에 새 logical dependency edge를 추가한다.

        방향:
            prerequisite_id -> dependent_id

        주의:
            이 함수는 구조적으로 명백히 잘못된 edge만 막는다.
            cycle 검사는 topological-sort 모듈에서 담당한다.
        """
        if edge.prerequisite_id not in self.V or edge.dependent_id not in self.V:
            raise ValueError("Edge contains node ids not present in V.")

        if edge.prerequisite_id == edge.dependent_id:
            raise ValueError("Self-loop is not allowed in Query Logic DAG.")

        # 중복 edge는 추가하지 않는다.
        for existing in self.E:
            if (
                existing.prerequisite_id == edge.prerequisite_id
                and existing.dependent_id == edge.dependent_id
            ):
                return

        # edge metadata가 비어 있으면 논문 기준 기본값을 채운다.
        edge.metadata.setdefault("relation_type", EDGE_RELATION_LOGICAL_PRECEDENCE)
        edge.metadata.setdefault("source", EDGE_SOURCE_LLM_DEPENDENCY_MODELING)

        self.E.append(edge)
        self.rebuild_indexes()

    def set_topological_order(self, order: List[int]) -> None:
        """
        topological-sort 모듈이 계산한 node 실행 순서를 저장한다.

        이 Builder는 edge를 만들 뿐, cycle 검증과 topological sort는 담당하지 않는다.
        """
        self.topological_order = order

    def set_ranks(self, ranks: Dict[int, int]) -> None:
        """
        graph pruning 또는 rank 계산 모듈이 계산한 node rank를 저장한다.

        논문에서는 같은 topological rank의 subproblem을 묶어
        unified query로 처리하는 graph pruning 전략을 설명한다.
        """
        self.ranks = ranks

    def mark_acyclic_verified(self, is_verified: bool = True) -> None:
        """
        DAG의 acyclicity 검증 여부를 metadata에 기록한다.

        주의:
            실제 cycle 검사는 여기서 수행하지 않는다.
            topological-sort / graph validation 모듈이 검증한 뒤 이 값을 업데이트한다.
        """
        self.metadata["is_acyclic_verified"] = is_verified

    def node_texts_in_id_order(self) -> List[str]:
        """
        기존 LogicRAG의 _topological_sort()와 연결하기 위한 임시 호환 함수.

        반환:
            node id 순서대로 정렬된 subproblem text 리스트.

        주의:
            논문 기준 최종 실행 순서는 topological sort 결과여야 한다.
            이 함수는 기존 코드와 연결하기 위해 text만 뽑아주는 helper다.
        """
        return [
            self.V[node_id].text
            for node_id in sorted(self.V.keys())
        ]

    def to_legacy_dependency_pairs(self) -> List[Tuple[int, int]]:
        """
        기존 코드의 dependency pair 형식으로 edge를 변환한다.

        새 DAG edge 방향:
            prerequisite_id -> dependent_id

        기존 코드 형식:
            (dependent_idx, dependency_idx)

        예:
            새 DAG edge: 0 -> 2
            기존 코드 pair: (2, 0)

        왜 필요한가:
            현재 repo의 기존 _topological_sort()가 아직 예전 pair 형식을 기대하기 때문이다.
            즉, 이 함수는 논문 구현의 핵심이라기보다는 기존 코드 연결용이다.
        """
        return [
            (edge.dependent_id, edge.prerequisite_id)
            for edge in self.E
        ]

    def to_dict(self) -> Dict[str, Any]:
        """
        DAG를 logging/evaluation용 dict로 변환한다.

        왜 필요한가:
            baseline 재현에서는 "어떤 Q에서 어떤 P가 만들어졌고,
            어떤 edge가 생겼는지"를 나중에 확인할 수 있어야 한다.
        """
        return {
            "metadata": self.metadata,
            "nodes": {
                node_id: {
                    "id": node.id,
                    "text": node.text,
                    "metadata": node.metadata,
                }
                for node_id, node in self.V.items()
            },
            "edges": [
                {
                    "prerequisite_id": edge.prerequisite_id,
                    "dependent_id": edge.dependent_id,
                    "reason": edge.reason,
                    "metadata": edge.metadata,
                }
                for edge in self.E
            ],
            "parents": {
                node_id: sorted(list(parent_ids))
                for node_id, parent_ids in self.parents.items()
            },
            "children": {
                node_id: sorted(list(child_ids))
                for node_id, child_ids in self.children.items()
            },
            "topological_order": self.topological_order,
            "ranks": self.ranks,
        }


class QueryLogicDAGBuilder:
    """
    Query Logic Dependency Graph G=(V,E)를 만드는 Builder.

    담당 범위:
        1. 분해된 subproblem P를 node 집합 V로 변환
        2. subproblem 사이의 logical precedence를 LLM으로 추론하여 edge 집합 E 생성
        3. 구조적으로 잘못된 edge 제거
           - 없는 node id를 참조하는 edge
           - self-loop
           - duplicate edge

    담당하지 않는 것:
        - retrieval 수행
        - final answer 생성
        - cycle 검사
        - topological sort
        - graph pruning
        - context pruning

    cycle 검사는 논문에서 topological sorting을 통해 검증한다고 되어 있으므로,
    이 Builder가 아니라 별도 graph validation / topological-sort 모듈에서 처리한다.
    """

    def construct_from_subproblems(
        self,
        question: str,
        subproblems: List[Dict[str, Any]],
    ) -> QueryLogicDAG:
        """
        논문 기준 메인 생성자.

        입력:
            question:
                원래 query Q.

            subproblems:
                query decomposition 결과 P.
                예:
                    [
                        {"id": 0, "text": "Find the country where the Eiffel Tower is located."},
                        {"id": 1, "text": "Find the capital of that country."}
                    ]

        처리:
            Q와 P를 받아서 Query Logic Dependency Graph G=(V,E)를 만든다.

        반환:
            QueryLogicDAG 객체.
        """
        nodes = self._build_nodes_from_subproblems(subproblems)

        raw_edges, edge_inference_status = self._infer_dependency_edges(
            question=question,
            nodes=nodes,
        )

        edges = self._validate_dependency_edges(
            nodes=nodes,
            edges=raw_edges,
        )

        dag = self._build_dag(
            question=question,
            nodes=nodes,
            edges=edges,
            construction_source="query_decomposition",
            edge_inference_status=edge_inference_status,
        )

        return dag

    def construct_from_dependency_texts(
        self,
        question: str,
        dependencies: List[str],
    ) -> QueryLogicDAG:
        """
        기존 코드 호환용 생성자.

        중요한 점:
            논문 기준으로는 Q를 subproblem P로 분해한 뒤
            construct_from_subproblems()를 호출하는 것이 맞다.

            하지만 현재 repo의 기존 LogicRAG는 아직 dependencies라는
            문자열 리스트를 넘길 수 있다. 그래서 이 함수는 그 리스트를
            임시 subproblem 형식으로 바꿔서 DAG를 만든다.

        즉:
            이 함수는 논문 정식 흐름이라기보다는
            기존 코드가 깨지지 않게 하기 위한 연결용 wrapper다.
        """
        subproblems = []

        for i, dependency_text in enumerate(dependencies):
            subproblems.append({
                "id": i,
                "text": dependency_text,
                "source": NODE_SOURCE_EXISTING_DEPENDENCY_TEXTS,
                "original_position": i,
                "created_by": NODE_CREATED_BY_WARM_UP_ANALYSIS,
            })

        nodes = self._build_nodes_from_subproblems(subproblems)

        raw_edges, edge_inference_status = self._infer_dependency_edges(
            question=question,
            nodes=nodes,
        )

        edges = self._validate_dependency_edges(
            nodes=nodes,
            edges=raw_edges,
        )

        dag = self._build_dag(
            question=question,
            nodes=nodes,
            edges=edges,
            construction_source="existing_dependency_texts",
            edge_inference_status=edge_inference_status,
        )

        return dag

    def _build_dag(
        self,
        question: str,
        nodes: Dict[int, SubproblemNode],
        edges: List[DependencyEdge],
        construction_source: str,
        edge_inference_status: str,
    ) -> QueryLogicDAG:
        """
        node 집합 V와 edge 집합 E를 QueryLogicDAG 객체로 묶는다.

        여기서 original_query를 DAG metadata에 저장한다.

        왜 original_query를 DAG에 저장하는가:
            원 질문 Q는 특정 node 하나의 정보가 아니라
            전체 graph가 어떤 질문에서 만들어졌는지를 나타내는 정보이기 때문이다.
        """
        dag = QueryLogicDAG(
            V=nodes,
            E=edges,
            metadata={
                "original_query": question,
                "graph_type": GRAPH_TYPE_QUERY_LOGIC_DEPENDENCY_GRAPH,
                "construction_source": construction_source,
                "edge_semantics": "logical_dependencies",
                "edge_inference_status": edge_inference_status,
                "is_acyclic_verified": False,
            },
        )

        dag.rebuild_indexes()
        return dag

    def _build_nodes_from_subproblems(
        self,
        subproblems: List[Dict[str, Any]],
    ) -> Dict[int, SubproblemNode]:
        """
        query decomposition 결과 P를 node 집합 V로 변환한다.

        논문 대응:
            p_i -> v_i

        검증:
            - subproblem은 dict여야 한다.
            - text는 비어 있으면 안 된다.
            - node id가 중복되면 안 된다.
        """
        nodes: Dict[int, SubproblemNode] = {}

        for i, sp in enumerate(subproblems):
            if not isinstance(sp, dict):
                raise ValueError(f"Subproblem must be a dict, got: {type(sp)}")

            node_id = int(sp.get("id", i))
            text = str(sp.get("text", "")).strip()

            if not text:
                raise ValueError(f"Subproblem {node_id} has empty text.")

            if node_id in nodes:
                raise ValueError(f"Duplicate subproblem id found: {node_id}")

            # subproblem dict에서 id/text를 제외한 나머지는 metadata로 보존한다.
            # 이렇게 하면 query decomposition 모듈이 추가 정보를 넘겨도 버리지 않는다.
            metadata = {
                key: value
                for key, value in sp.items()
                if key not in {"id", "text"}
            }

            # 논문 기준 기본 node metadata.
            # 이미 외부에서 값이 들어온 경우에는 덮어쓰지 않는다.
            metadata.setdefault("source", NODE_SOURCE_INITIAL_DECOMPOSITION)
            metadata.setdefault("original_position", i)
            metadata.setdefault("created_by", NODE_CREATED_BY_QUERY_DECOMPOSITION)

            nodes[node_id] = SubproblemNode(
                id=node_id,
                text=text,
                metadata=metadata,
            )

        return nodes

    def _infer_dependency_edges(
        self,
        question: str,
        nodes: Dict[int, SubproblemNode],
    ) -> Tuple[List[DependencyEdge], str]:
        """
        LLM을 사용해 subproblem 사이의 logical dependency edge를 추론한다.

        논문 대응:
            Dependency Modeling 단계.
            LLM이 subproblem 사이의 logical precedence를 기준으로 edge를 만든다.

        edge 방향:
            prerequisite_id -> dependent_id

        중요한 규칙:
            두 subproblem이 비슷한 주제라고 해서 edge를 만들면 안 된다.
            반드시 "앞 subproblem의 결과가 뒤 subproblem을 푸는 데 필요할 때"만 edge를 만든다.
        """
        nodes_payload = [
            {
                "id": node.id,
                "text": node.text,
            }
            for node in nodes.values()
        ]

        prompt = f"""
You are constructing the edge set E of a Query Logic Dependency Graph G=(V,E).

Original query Q:
{question}

Subproblem nodes V:
{json.dumps(nodes_payload, ensure_ascii=False, indent=2)}

Task:
Infer directed logical dependency edges among the subproblem nodes.

Paper-grounded definition:
- Each node v_i represents a subproblem p_i.
- A directed edge A -> B means subproblem A is a logical prerequisite of subproblem B.
- In other words, A must be resolved before B because B depends on the result, entity, date, comparison target, or intermediate conclusion produced by A.

Use this exact direction:
prerequisite_id -> dependent_id

Rules:
- Use only ids from the provided subproblem nodes.
- Do not create self-loops.
- Do not include independent nodes in edges.
- Do not create an edge merely because two subproblems are topically related.
- Create an edge only when there is logical precedence.
- Prefer direct logical dependencies only.
- Do not add redundant transitive edges.
  For example, if 0 -> 1 and 1 -> 2, do not also add 0 -> 2 unless it is directly necessary.
- Each edge must include a short reason.
- If no logical dependencies exist, return an empty edge list.

Few-shot example:
Original query:
Which film was released earlier, Inception or Titanic?

Subproblem nodes V:
[
  {{"id": 0, "text": "Find the release date of Inception."}},
  {{"id": 1, "text": "Find the release date of Titanic."}},
  {{"id": 2, "text": "Compare the two release dates and determine which film was released earlier."}}
]

Output:
{{
  "edges": [
    {{
      "prerequisite_id": 0,
      "dependent_id": 2,
      "reason": "The release date of Inception is needed before comparing the two films."
    }},
    {{
      "prerequisite_id": 1,
      "dependent_id": 2,
      "reason": "The release date of Titanic is needed before comparing the two films."
    }}
  ]
}}

Return ONLY a JSON object with this schema:
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
        except APIError as e:
            logger.error(
                "Dependency edge inference APIError %s: %s",
                e.__class__.__name__,
                e,
            )
            return [], "api_error"

        response = response.strip()
        response = response.replace("```json", "").replace("```", "")

        result = fix_json_response(response)

        if not isinstance(result, dict):
            logger.warning("Failed to parse dependency edge inference response.")
            return [], "invalid_output"

        if "edges" not in result:
            logger.warning("Dependency edge inference result missing 'edges'.")
            return [], "invalid_output"

        raw_edges = result["edges"]

        if not isinstance(raw_edges, list):
            logger.warning("Dependency edge inference result has non-list 'edges'.")
            return [], "invalid_output"

        edges: List[DependencyEdge] = []

        for raw_edge in raw_edges:
            try:
                prerequisite_id = int(raw_edge["prerequisite_id"])
                dependent_id = int(raw_edge["dependent_id"])
                reason = str(raw_edge.get("reason", "")).strip()

                # 논문 기준 edge metadata.
                # LLM prompt에는 metadata를 요구하지 않고,
                # Builder가 논문 의미에 맞게 기본 metadata를 부여한다.
                metadata = {
                    "relation_type": EDGE_RELATION_LOGICAL_PRECEDENCE,
                    "source": EDGE_SOURCE_LLM_DEPENDENCY_MODELING,
                    "created_by": EDGE_CREATED_BY_INITIAL_GRAPH_CONSTRUCTION,
                }

                edges.append(
                    DependencyEdge(
                        prerequisite_id=prerequisite_id,
                        dependent_id=dependent_id,
                        reason=reason,
                        metadata=metadata,
                    )
                )

            except Exception as e:
                logger.warning(f"Skipping malformed dependency edge: {raw_edge}. Error: {e}")
                continue

        return edges, "ok"

    def _validate_dependency_edges(
        self,
        nodes: Dict[int, SubproblemNode],
        edges: List[DependencyEdge],
    ) -> List[DependencyEdge]:
        """
        LLM이 만든 edge 중 구조적으로 잘못된 edge를 제거한다.

        제거 대상:
            1. 존재하지 않는 node id를 가리키는 edge
            2. 자기 자신을 가리키는 self-loop
            3. 중복 edge

        주의:
            cycle 검사는 여기서 하지 않는다.
            논문에서 graph acyclicity는 topological sort를 통해 검증된다고 되어 있으므로,
            cycle 검사는 별도 topological-sort / graph validation 모듈에서 처리한다.
        """
        node_ids = set(nodes.keys())
        validated: List[DependencyEdge] = []
        seen = set()

        for edge in edges:
            try:
                pre = int(edge.prerequisite_id)
                dep = int(edge.dependent_id)
            except Exception:
                logger.warning(f"Skipping edge with malformed ids: {edge}")
                continue

            if pre not in node_ids or dep not in node_ids:
                logger.warning(
                    f"Skipping edge with unknown node id: "
                    f"prerequisite_id={pre}, dependent_id={dep}"
                )
                continue

            if pre == dep:
                logger.warning(f"Skipping self-loop edge: {pre} -> {dep}")
                continue

            key = (pre, dep)
            if key in seen:
                logger.warning(f"Skipping duplicate edge: {pre} -> {dep}")
                continue

            edge.metadata.setdefault("relation_type", EDGE_RELATION_LOGICAL_PRECEDENCE)
            edge.metadata.setdefault("source", EDGE_SOURCE_LLM_DEPENDENCY_MODELING)

            seen.add(key)
            validated.append(
                DependencyEdge(
                    prerequisite_id=pre,
                    dependent_id=dep,
                    reason=edge.reason.strip() if edge.reason else "",
                    metadata=edge.metadata,
                )
            )

        return validated
