"""
** non-cyclicity verification 유틸리티 **

- QueryLogicDAG의 edge set E를 검증하고, 생성된 의존성 관계가 순환하지 않는지 확인
- QueryLogicDAG.V / QueryLogicDAG.E를 Kahn/DFS용 index graph로 정규화
- Kahn algorithm으로 partial order / blocked node 계산
- 3-state DFS로 cycle을 탐지
- 정규화된 graph가 DAG인 경우 topological order와 sorted_dependencies 반환
- fallback에 필요한 metadata를 명시적으로 제공

DAG 입력 형식:
    QueryLogicDAG
        V: Dict[int, SubproblemNode]
        E: List[DependencyEdge]
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


@dataclass
class DAGVerificationResult:
    """검증 결과를 담는 class"""
    is_dag: bool    # DAG 여부
    has_cycle: bool # cycle 존재 여부

    # ----------------------------------------------
    # 전체 입력 node 정보
    # ----------------------------------------------
    input_node_ids: List[int] = field(default_factory=list) # node id 목록
    input_dependencies: List[str] = field(default_factory=list) # subproblem text 목록
    node_id_to_index: Dict[int, int] = field(default_factory=dict)  # 실제 DAG node id를 Kahn/DFS에서 쓸 내부 index로 바꾼 mapping (index 혼동방지)
    index_to_node_id: Dict[int, int] = field(default_factory=dict)  # 내부 index를 다시 실제 DAG node id로 되돌리는 mapping

    # ----------------------------------------------
    # cycle이 없을 때 최종 topological sort 결과
    # ----------------------------------------------
    sorted_indices: List[int] = field(default_factory=list) # Kahn/DFS 내부 index 기준의 최종 정렬 순서
    sorted_node_ids: List[int] = field(default_factory=list)    # 실제 DAG node id 기준의 최종 정렬 순서
    sorted_dependencies: List[str] = field(default_factory=list)    # subproblem text 기준의 최종 정렬 결과
    
    # ----------------------------------------------
    # Kahn 알고리즘 기준 partial order 
    # - cycle이 없으면 -> sorted_indices와 같음
    # - cycle이 있으면 -> cycle에 막히기 전까지 처리 가능한 prefix만 들어감
    # ----------------------------------------------   
    partial_order_indices: List[int] = field(default_factory=list)
    partial_sorted_dependencies: List[str] = field(default_factory=list)
    partial_order_node_ids: List[int] = field(default_factory=list)

    # ----------------------------------------------
    # cycle 때문에 처리되지 못한 node 정보
    # - Kahn 알고리즘에서 indegree가 끝까지 0이 되지 않아 처리 못 한 내부 index 목록
    # ----------------------------------------------
    blocked_indices: List[int] = field(default_factory=list)
    blocked_node_ids: List[int] = field(default_factory=list)
    blocked_dependencies: List[str] = field(default_factory=list)

    # ----------------------------------------------
    # 정상 edge / invalid edge / duplicate edge 정보
    # ----------------------------------------------
    valid_dependency_pairs: List[List[int]] = field(default_factory=list)   # 검증을 통과한 edge를 기존 pair 스타일로 저장한 값 [dependent_internal_idx, prerequisite_internal_idx]
    valid_dependency_edges: List[Dict[str, Any]] = field(default_factory=list)
    invalid_edges: List[Dict[str, Any]] = field(default_factory=list)   # 구조적으로 잘못되어 graph에 넣지 않은 edge 목록
    duplicate_edges: List[Dict[str, Any]] = field(default_factory=list) # 중복이라서 graph에 한 번만 넣고 나머지는 무시한 edge 목록

    # ----------------------------------------------
    # DFS cycle detection 결과
    # ----------------------------------------------
    cycle_indices: List[int] = field(default_factory=list)
    cycle_node_ids: List[int] = field(default_factory=list)
    cycle_dependencies: List[str] = field(default_factory=list) # cycle을 구성하는 경로

    # ----------------------------------------------
    # 입력 검증 상태와 메시지
    # ----------------------------------------------
    input_is_valid: bool = True # 입력 형식 검증 (pair 형식, index 정수, 범위 이탈, self-loop 등)
    has_invalid_edges: bool = False # invalid edge가 하나라도 있었는지 여부
    message: str = ""   # 검증 결과 요약 메시지

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DependencyGraphCycleError(ValueError):
    """ 그래프에 cycle이 있어 valid DAG를 만들 수 없을 때 발생하는 예외 메세지 """
    def __init__(self, dag_result: Dict[str, Any]):
        self.dag_result = dag_result

        cycle_node_ids = dag_result.get("cycle_node_ids", [])
        cycle_dependencies = dag_result.get("cycle_dependencies", [])
        invalid_edges = dag_result.get("invalid_edges", [])

        message = (
            "Dependency graph is not a valid DAG. "
            f"cycle_node_ids={cycle_node_ids}, "
            f"cycle_dependencies={cycle_dependencies}, "
            f"invalid_edges={invalid_edges}"
        )

        super().__init__(message)
    
def _edge_to_dict(edge: Any) -> Dict[str, Any]:
    """ DependencyEdge를 logging 가능한 dict 형식으로 변환 """
    if not isinstance(edge, dict):
        return {
            "prerequisite_id": None,
            "dependent_id": None,
            "reason": "",
            "raw_edge": repr(edge),
        }

    return {
        "prerequisite_id": edge.get("prerequisite_id"),
        "dependent_id": edge.get("dependent_id"),
        "reason": edge.get("reason", ""),
    }

def _normalize_query_logic_dag(
    dag: Any,
) -> Tuple[
    List[str],                  # dependencies, internal index order
    List[int],                  # input_node_ids, internal index -> node_id
    Dict[int, int],             # node_id_to_index
    Dict[int, int],             # index_to_node_id
    List[Tuple[int, int]],      # valid_pairs, old-compatible [dependent_idx, prerequisite_idx]
    List[Dict[str, Any]],       # valid_edges, node-id based
    List[Dict[str, Any]],       # invalid_edges
    List[Dict[str, Any]],       # duplicate_edges
    Dict[int, List[int]],       # graph, prerequisite_idx -> dependent_idx
]:
    """JSON/dict DAG 입력을 기존 Kahn/DFS 알고리즘용 index graph로 정규화"""

    # ------------------------------------------------------------
    # 1. 입력 DAG 형식 확인
    # ------------------------------------------------------------
    if not isinstance(dag, dict):
        invalid_edges = [{
            "edge": repr(dag),
            "reason": "dag_input_must_be_dict",
        }]
        return [], [], {}, {}, [], [], invalid_edges, [], {}

    if "V" in dag and "E" in dag:
        raw_nodes = dag["V"]
        raw_edges = dag["E"]
        node_field_name = "V"
        edge_field_name = "E"

    elif "nodes" in dag and "edges" in dag:
        raw_nodes = dag["nodes"]
        raw_edges = dag["edges"]
        node_field_name = "nodes"
        edge_field_name = "edges"

    else:
        invalid_edges = [{
            "edge": repr(dag),
            "reason": "dag_dict_must_have_either_V_E_or_nodes_edges",
        }]
        return [], [], {}, {}, [], [], invalid_edges, [], {}

    if not isinstance(raw_nodes, dict):
        invalid_edges = [{
            "edge": repr(raw_nodes),
            "reason": f"dag_{node_field_name}_must_be_dict",
        }]
        return [], [], {}, {}, [], [], invalid_edges, [], {}

    if not isinstance(raw_edges, list):
        invalid_edges = [{
            "edge": repr(raw_edges),
            "reason": f"dag_{edge_field_name}_must_be_list",
        }]
        return [], [], {}, {}, [], [], invalid_edges, [], {}

    # ------------------------------------------------------------
    # 2. V에서 node id와 text 추출
    # ------------------------------------------------------------
    node_text_by_id: Dict[int, str] = {}
    invalid_edges: List[Dict[str, Any]] = []

    for raw_key, raw_node in raw_nodes.items():
        if not isinstance(raw_node, dict):
            invalid_edges.append({
                "edge": {
                    "node_key": raw_key,
                    "node": repr(raw_node),
                },
                "reason": "node_must_be_dict",
            })
            continue

        raw_node_id = raw_node.get("id", raw_key)

        if isinstance(raw_node_id, bool):
            invalid_edges.append({
                "edge": {
                    "node_key": raw_key,
                    "node": raw_node,
                },
                "reason": "node_id_must_be_integer",
            })
            continue

        try:
            node_id = int(raw_node_id)
        except (TypeError, ValueError):
            invalid_edges.append({
                "edge": {
                    "node_key": raw_key,
                    "node": raw_node,
                },
                "reason": "node_id_must_be_integer",
            })
            continue

        text = raw_node.get("text")

        if not isinstance(text, str):
            invalid_edges.append({
                "edge": {
                    "node_key": raw_key,
                    "node": raw_node,
                },
                "reason": "node_text_must_be_string",
            })
            continue

        if node_id in node_text_by_id:
            invalid_edges.append({
                "edge": {
                    "node_key": raw_key,
                    "node": raw_node,
                },
                "reason": "duplicate_node_id",
            })
            continue

        node_text_by_id[node_id] = text

    # valid node가 하나도 없으면 더 진행할 수 없음
    if not node_text_by_id:
        return [], [], {}, {}, [], [], invalid_edges, [], {}

    # ------------------------------------------------------------
    # 3. 실제 node id와 내부 index mapping 생성
    # ------------------------------------------------------------
    input_node_ids = sorted(node_text_by_id.keys())

    node_id_to_index = {
        node_id: idx
        for idx, node_id in enumerate(input_node_ids)
    }

    index_to_node_id = {
        idx: node_id
        for node_id, idx in node_id_to_index.items()
    }

    dependencies = [
        node_text_by_id[node_id]
        for node_id in input_node_ids
    ]

    num_dependencies = len(dependencies)

    graph: Dict[int, List[int]] = {
        idx: []
        for idx in range(num_dependencies)
    }

    # ------------------------------------------------------------
    # 4. E에서 edge 추출 후 graph 생성
    # ------------------------------------------------------------
    valid_pairs: List[Tuple[int, int]] = []
    valid_edges: List[Dict[str, Any]] = []
    duplicate_edges: List[Dict[str, Any]] = []
    seen_graph_edges = set()

    for raw_edge in raw_edges:
        edge_log = _edge_to_dict(raw_edge)

        if not isinstance(raw_edge, dict):
            invalid_edges.append({
                "edge": edge_log,
                "reason": "edge_must_be_dict",
            })
            continue

        prerequisite_id = edge_log["prerequisite_id"]
        dependent_id = edge_log["dependent_id"]

        if (
            not isinstance(prerequisite_id, int)
            or isinstance(prerequisite_id, bool)
            or not isinstance(dependent_id, int)
            or isinstance(dependent_id, bool)
        ):
            invalid_edges.append({
                "edge": edge_log,
                "reason": "node_ids_must_be_integers",
            })
            continue

        if prerequisite_id not in node_id_to_index or dependent_id not in node_id_to_index:
            invalid_edges.append({
                "edge": edge_log,
                "reason": "node_id_not_found_in_dag_nodes",
            })
            continue

        if prerequisite_id == dependent_id:
            invalid_edges.append({
                "edge": edge_log,
                "reason": "self_loop",
            })
            continue

        prerequisite_idx = node_id_to_index[prerequisite_id]
        dependent_idx = node_id_to_index[dependent_id]

        graph_edge = (prerequisite_idx, dependent_idx)

        if graph_edge in seen_graph_edges:
            duplicate_edges.append(edge_log)
            continue

        seen_graph_edges.add(graph_edge)

        valid_pairs.append((dependent_idx, prerequisite_idx))
        valid_edges.append(edge_log)
        graph[prerequisite_idx].append(dependent_idx)

    return (
        dependencies,   # subproblem text list
        input_node_ids, # DAG node id 목록
        node_id_to_index,   # 내부 index 목록
        index_to_node_id,
        valid_pairs,    # dependency pair
        valid_edges,
        invalid_edges,  # 잘못되어 graph에 넣지 않은 edge
        duplicate_edges,    # 중복이라 graph에 한 번만 넣고 무시한 edge
        graph,  # Kahn/DFS가 실제로 사용할 내부 graph
    )


def _kahn_partial_order(graph: Dict[int, List[int]], num_dependencies: int) -> Tuple[List[int], List[int]]:
    """
    cycle에 의해 진행이 막히기 전까지 처리 가능한 node 순서 계산

    1) DAG인 경우(cycle X) : processed = 전체 topological order
    2) cycle이 있는 경우 : processsed = 먼저 처리 가능한 acyclic prefix, 
    blocked = cycle 또는 cycle의 영향 때문에 처리되지 못한 node 목록
    """
    indegree = [0] * num_dependencies
    for node in range(num_dependencies):
        for neighbor in graph[node]:
            indegree[neighbor] += 1

    queue: Deque[int] = deque(idx for idx in range(num_dependencies) if indegree[idx] == 0)
    processed: List[int] = []

    while queue:
        node = queue.popleft()
        processed.append(node)
        for neighbor in graph[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    processed_set = set(processed)
    blocked = [idx for idx in range(num_dependencies) if idx not in processed_set]
    return processed, blocked


def _dfs_cycle_and_topological_order(
    graph: Dict[int, List[int]],
    num_dependencies: int,
) -> Tuple[bool, List[int], List[int]]:
    """
    3-state DFS로 directed cycle을 탐지하고, DFS 기반 topological order를 계산

    - 0 = UNVISITED: 아직 방문하지 않음
    - 1 = VISITING: 현재 DFS 경로 위에 있음
    - 2 = DONE: 방문 완료
    """

    UNVISITED, VISITING, DONE = 0, 1, 2
    state = [UNVISITED] * num_dependencies
    parent = [-1] * num_dependencies
    postorder: List[int] = []
    cycle_indices: List[int] = []

    def reconstruct_cycle(current: int, neighbor: int) -> List[int]:
        path = [neighbor]
        node = current
        while node != neighbor and node != -1:
            path.append(node)
            node = parent[node]
        path.append(neighbor)
        path.reverse()
        return path

    def dfs(node: int) -> bool:
        nonlocal cycle_indices
        state[node] = VISITING

        for neighbor in graph[node]:
            if state[neighbor] == UNVISITED:
                parent[neighbor] = node
                if dfs(neighbor):
                    return True
            elif state[neighbor] == VISITING:
                cycle_indices = reconstruct_cycle(node, neighbor)
                return True

        state[node] = DONE
        postorder.append(node)
        return False

    for node in range(num_dependencies):
        if state[node] == UNVISITED and dfs(node):
            return True, cycle_indices, []

    return False, [], postorder[::-1]

def _texts_by_indices(dependencies: List[str], indices: List[int]) -> List[str]:
    return [dependencies[idx] for idx in indices]

def _node_ids_by_indices(index_to_node_id: Dict[int, int], indices: List[int]) -> List[int]:
    return [index_to_node_id[idx] for idx in indices]

def verify_dag_and_topological_sort(dag: Any) -> Dict[str, Any]:
    """
    LogicRAG dependency의 비순환성을 검증하고 topological order를 계산
    (기존 : dependencies + dependency_pairs 입력 -> 변경 : dag 하나를 받음)
    """
    (
        dependencies,
        input_node_ids,
        node_id_to_index,
        index_to_node_id,
        valid_pairs,
        valid_edges,
        invalid_edges,
        duplicate_edges,
        graph,
    ) = _normalize_query_logic_dag(dag)

    num_dependencies = len(dependencies)

    # 입력 자체가 QueryLogicDAG 형태가 아니거나 V가 비정상인 경우.
    if num_dependencies == 0 and invalid_edges:
        result = DAGVerificationResult(
            is_dag=False,
            has_cycle=False,
            input_node_ids=input_node_ids,
            input_dependencies=dependencies,
            node_id_to_index=node_id_to_index,
            index_to_node_id=index_to_node_id,
            valid_dependency_pairs=[list(pair) for pair in valid_pairs],
            valid_dependency_edges=valid_edges,
            invalid_edges=invalid_edges,
            duplicate_edges=duplicate_edges,
            input_is_valid=False,
            has_invalid_edges=True,
            message="Invalid QueryLogicDAG input.",
        )
        return result.to_dict()

    partial_order_indices, blocked_indices = _kahn_partial_order(graph, num_dependencies)
    has_cycle, cycle_indices, dfs_sorted_indices = _dfs_cycle_and_topological_order(
        graph,
        num_dependencies,
    )

    if has_cycle:
        result = DAGVerificationResult(
            is_dag=False,
            has_cycle=True,
            input_node_ids=input_node_ids,
            input_dependencies=dependencies,
            node_id_to_index=node_id_to_index,
            index_to_node_id=index_to_node_id,
            partial_order_indices=partial_order_indices,
            partial_order_node_ids=_node_ids_by_indices(index_to_node_id, partial_order_indices),
            partial_sorted_dependencies=_texts_by_indices(dependencies, partial_order_indices),
            blocked_indices=blocked_indices,
            blocked_node_ids=_node_ids_by_indices(index_to_node_id, blocked_indices),
            blocked_dependencies=_texts_by_indices(dependencies, blocked_indices),
            valid_dependency_pairs=[list(pair) for pair in valid_pairs],
            valid_dependency_edges=valid_edges,
            invalid_edges=invalid_edges,
            duplicate_edges=duplicate_edges,
            cycle_indices=cycle_indices,
            cycle_node_ids=_node_ids_by_indices(index_to_node_id, cycle_indices),
            cycle_dependencies=_texts_by_indices(dependencies, cycle_indices),
            input_is_valid=len(invalid_edges) == 0,
            has_invalid_edges=bool(invalid_edges),
            message="Cycle detected in dependency graph.",
        )
        return result.to_dict()

    sorted_indices = partial_order_indices
    sorted_dependencies = _texts_by_indices(dependencies, sorted_indices)

    if invalid_edges:
        message = "Valid DAG after removing structurally invalid edges."
    else:
        message = "Valid DAG."

    result = DAGVerificationResult(
        is_dag=True,
        has_cycle=False,
        input_node_ids=input_node_ids,
        input_dependencies=dependencies,
        node_id_to_index=node_id_to_index,
        index_to_node_id=index_to_node_id,
        sorted_indices=sorted_indices,
        sorted_node_ids=_node_ids_by_indices(index_to_node_id, sorted_indices),
        sorted_dependencies=sorted_dependencies,
        partial_order_indices=sorted_indices,
        partial_order_node_ids=_node_ids_by_indices(index_to_node_id, sorted_indices),
        partial_sorted_dependencies=sorted_dependencies,
        blocked_indices=[],
        blocked_node_ids=[],
        blocked_dependencies=[],
        valid_dependency_pairs=[list(pair) for pair in valid_pairs],
        valid_dependency_edges=valid_edges,
        invalid_edges=invalid_edges,
        duplicate_edges=duplicate_edges,
        cycle_indices=[],
        cycle_node_ids=[],
        cycle_dependencies=[],
        input_is_valid=len(invalid_edges) == 0,
        has_invalid_edges=bool(invalid_edges),
        message=message,
    )
    return result.to_dict()

def build_partial_order_fallback(dag_result: Dict[str, Any]) -> List[str]:
    """
    cycle이 있을 때 fallback 순서를 만드는 함수

    1. partial_order_indices에 있는 dependency를 먼저 넣는다.
    2. 아직 사용되지 않은 dependency를 원래 순서대로 뒤에 붙인다.
    """
    dependencies = dag_result.get("input_dependencies", []) or []
    partial_indices = dag_result.get("partial_order_indices", []) or []

    used = set(partial_indices)
    fallback_indices = list(partial_indices) + [
        idx for idx in range(len(dependencies)) if idx not in used
    ]
    return [dependencies[idx] for idx in fallback_indices]


def topological_sort_or_raise(dag: Any) -> List[str]:
    """그래프에 cycle이 있으면 예외를 발생시키는 헬퍼 함수."""
    dag_result = verify_dag_and_topological_sort(dag)
    if not dag_result["is_dag"]:
        raise DependencyGraphCycleError(dag_result)
    return dag_result["sorted_dependencies"]