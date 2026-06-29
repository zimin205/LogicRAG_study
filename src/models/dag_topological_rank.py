"""
** 검증한 DAG 결과를 받아서 topological rank를 계산하는 유틸리티 **

- dag_result를 입력으로 받아서 node별 rank 계산
- rank별 node/dependency group 생성
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Tuple


class TopologicalRankError(ValueError):
    """rank 계산이 불가능할 때 던지는 error"""


def compute_topological_ranks_from_verification(
    dag_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Input : dag_result

    Preconditions:
        dag_result["is_dag"] must be True.

    Returns:
        {
            "ranks": {node_id: rank},
            "rank_groups": {rank: [node_id, ...]},
            "ranked_dependencies": {rank: [subproblem_text, ...]},
            "max_rank": int,
            "topological_order_node_ids": [...],
            "topological_order_dependencies": [...],
        }
    """

    # 1. DAG 여부 확인
    if not dag_result.get("is_dag", False):
        raise TopologicalRankError(
            "Topological ranks can only be computed for a valid DAG. "
            "Run cycle repair or use a valid verified DAG first."
        )

    # 2. verifier 결과에서 정보 가져오기
    node_ids: List[int] = list(dag_result.get("input_node_ids", []) or [])
    dependencies: List[str] = list(dag_result.get("input_dependencies", []) or [])
    edges: List[Dict[str, Any]] = list(dag_result.get("valid_dependency_edges", []) or [])

    if len(node_ids) != len(dependencies):
        raise TopologicalRankError(
            "Invalid dag_result: input_node_ids and input_dependencies length mismatch."
        )

    node_id_set = set(node_ids)
    node_text_by_id = {
        node_id: dependencies[idx]
        for idx, node_id in enumerate(node_ids)
    }

    graph = {node_id: [] for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    seen_edges = set()

    for edge in edges:
        if not isinstance(edge, dict):
            continue

        pre = edge.get("prerequisite_id")
        dep = edge.get("dependent_id")

        if pre not in node_id_set or dep not in node_id_set:
            continue

        if pre == dep:
            continue

        edge_key = (pre, dep)
        if edge_key in seen_edges:
            continue

        seen_edges.add(edge_key)
        graph[pre].append(dep)
        indegree[dep] += 1

    # 3. Kahn algorithm 확장으로 rank 계산
    queue = deque(sorted(node_id for node_id in node_ids if indegree[node_id] == 0))
    ranks = {node_id: 0 for node_id in node_ids}
    processed = []

    while queue:
        node = queue.popleft()
        processed.append(node)

        for child in graph[node]:
            ranks[child] = max(ranks[child], ranks[node] + 1)
            indegree[child] -= 1

            if indegree[child] == 0:
                queue.append(child)

    if len(processed) != len(node_ids):
        raise TopologicalRankError(
            "Rank computation detected an unresolved cycle or malformed DAG result."
        )

    # 4. rank group 만들기
    rank_groups: Dict[int, List[int]] = {}
    ranked_dependencies: Dict[int, List[str]] = {}

    for node_id in sorted(node_ids, key=lambda nid: (ranks[nid], nid)):
        rank = ranks[node_id]
        rank_groups.setdefault(rank, []).append(node_id)
        ranked_dependencies.setdefault(rank, []).append(node_text_by_id[node_id])

    # max_rank 계산
    max_rank = max(ranks.values()) if ranks else 0

    return {
        "ranks": ranks,
        "rank_groups": rank_groups,
        "ranked_dependencies": ranked_dependencies,
        "max_rank": max_rank,
        "topological_order_node_ids": dag_result.get("sorted_node_ids", []) or [],
        "topological_order_dependencies": dag_result.get("sorted_dependencies", []) or [],
    }


def attach_topological_ranks_to_dag_dict(
    dag_dict: Dict[str, Any],
    rank_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    rank 결과를 DAG dict에 붙여주는 헬퍼 함수
    """

    enriched = dict(dag_dict)

    enriched["ranks"] = rank_result.get("ranks", {})
    enriched["rank_groups"] = rank_result.get("rank_groups", {})
    enriched["ranked_dependencies"] = rank_result.get("ranked_dependencies", {})
    enriched["max_rank"] = rank_result.get("max_rank", 0)

    return enriched