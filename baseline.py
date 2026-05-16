"""Baseline route generation for the Routio experiment."""

from __future__ import annotations

import warnings

import networkx as nx

from graph_utils import edge_key, route_edges
from metrics import evaluate_network


MAX_ROUTE_SIMILARITY = 0.70
OVERLAP_WEIGHT = 0.8
SHORT_ROUTE_WEIGHT = 3.0
LENGTH_WEIGHT = 1.0
DISTRICT_BONUS_WEIGHT = 0.0


def rank_od_pairs(od_matrix, district_ids, include_diagonal=False):
    """Return district pairs sorted by descending OD demand."""
    pairs = []

    for from_idx, from_district in enumerate(district_ids):
        for to_idx, to_district in enumerate(district_ids):
            if not include_diagonal and from_idx == to_idx:
                continue
            pairs.append(
                {
                    "from_district": from_district,
                    "to_district": to_district,
                    "demand": od_matrix[from_idx][to_idx],
                }
            )

    return sorted(pairs, key=lambda item: item["demand"], reverse=True)


def rank_undirected_od_pairs(od_matrix, district_ids):
    """Return district pairs ranked by combined two-way OD demand."""
    pairs = []

    for from_idx, from_district in enumerate(district_ids):
        for to_idx in range(from_idx + 1, len(district_ids)):
            to_district = district_ids[to_idx]
            demand = od_matrix[from_idx][to_idx] + od_matrix[to_idx][from_idx]
            pairs.append(
                {
                    "from_district": from_district,
                    "to_district": to_district,
                    "demand": demand,
                }
            )

    return sorted(pairs, key=lambda item: item["demand"], reverse=True)


def choose_representative_stop_pair(graph, stops_from, stops_to):
    """Choose the stop pair with the shortest weighted graph distance."""
    best_pair = None
    best_length = None

    for source in stops_from:
        for target in stops_to:
            if source == target:
                continue
            try:
                path_length = nx.shortest_path_length(
                    graph,
                    source=source,
                    target=target,
                    weight="weight",
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            if best_length is None or path_length < best_length:
                best_pair = (source, target)
                best_length = path_length

    return best_pair


def choose_representative_route(graph, stops_from, stops_to, existing_routes, config):
    """Choose a useful shortest-path route, not just the closest stop pair."""
    best_route = None
    best_score = None

    for source in stops_from:
        for target in stops_to:
            if source == target:
                continue
            try:
                path = build_shortest_path_route(graph, source, target)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            if len(path) > config["max_route_length"]:
                continue

            route = extend_route_to_min_length(graph, path, existing_routes, config)
            if is_too_similar(route, existing_routes):
                continue

            score = route_score(graph, route, existing_routes, config)
            if best_score is None or score < best_score:
                best_route = route
                best_score = score

    return best_route


def build_shortest_path_route(graph, source_stop, target_stop):
    """Build one baseline route as a weighted shortest path."""
    return nx.shortest_path(
        graph,
        source=source_stop,
        target=target_stop,
        weight="weight",
    )


def build_baseline_routes(graph, district_ids, od_matrix, stops_by_district, config):
    """Build deterministic greedy baseline routes from high-demand OD pairs."""
    routes = []

    for pair in rank_undirected_od_pairs(od_matrix, district_ids):
        if len(routes) >= config["num_routes"]:
            break

        from_district = pair["from_district"]
        to_district = pair["to_district"]
        stops_from = stops_by_district.get(from_district, [])
        stops_to = stops_by_district.get(to_district, [])

        if not stops_from or not stops_to:
            warnings.warn(
                f"Skipping OD pair {from_district!r} -> {to_district!r}: no stops in one of the districts.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        route = choose_representative_route(
            graph,
            stops_from,
            stops_to,
            routes,
            config,
        )
        if route is None:
            warnings.warn(
                f"Skipping OD pair {from_district!r} - {to_district!r}: no suitable baseline route.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if len(route) >= 2:
            routes.append(route)

    return routes


def extend_route_to_min_length(graph, route, existing_routes, config):
    """Greedily extend a short route at either end without repeating stops."""
    route = list(route)
    min_length = config["min_route_length"]
    max_length = config["max_route_length"]

    while len(route) < min_length and len(route) < max_length:
        candidates = extension_candidates(graph, route)
        if not candidates:
            break

        route = min(
            candidates,
            key=lambda candidate: route_score(graph, candidate, existing_routes, config),
        )

    return route


def extension_candidates(graph, route):
    """Return one-stop extensions on the left and right sides of a route."""
    visited = set(route)
    candidates = []

    for neighbor in graph.neighbors(route[0]):
        if neighbor not in visited:
            candidates.append([neighbor, *route])

    for neighbor in graph.neighbors(route[-1]):
        if neighbor not in visited:
            candidates.append([*route, neighbor])

    return candidates


def route_score(graph, route, existing_routes, config):
    """Score a baseline candidate; lower is better."""
    route_edges_count = max(1, len(route) - 1)
    min_length = config["min_route_length"]
    max_length = config["max_route_length"]
    too_short = max(0, min_length - len(route)) / min_length
    too_long = max(0, len(route) - max_length) / max_length
    overlap = route_overlap_ratio(route, existing_routes)
    length = route_weight(graph, route)
    length_normalizer = max_graph_edge_weight(graph) * max_length
    normalized_length = length / length_normalizer if length_normalizer else 0.0
    district_bonus = max(0, len(route_districts(graph, route)) - 2) / route_edges_count

    return (
        SHORT_ROUTE_WEIGHT * too_short
        + SHORT_ROUTE_WEIGHT * too_long
        + OVERLAP_WEIGHT * overlap
        + LENGTH_WEIGHT * normalized_length
        - DISTRICT_BONUS_WEIGHT * district_bonus
    )


def route_overlap_ratio(route, existing_routes):
    """Return the maximum share of candidate edges already used by existing routes."""
    candidate_edges = route_edge_set(route)
    if not candidate_edges:
        return 0.0

    existing_edges = set()
    for existing_route in existing_routes:
        existing_edges.update(route_edge_set(existing_route))

    if not existing_edges:
        return 0.0

    return len(candidate_edges & existing_edges) / len(candidate_edges)


def is_too_similar(route, existing_routes, threshold=MAX_ROUTE_SIMILARITY):
    """Return True when a route is nearly a duplicate of an existing route."""
    candidate_edges = route_edge_set(route)
    if not candidate_edges:
        return True

    for existing_route in existing_routes:
        existing_edges = route_edge_set(existing_route)
        if not existing_edges:
            continue
        similarity = len(candidate_edges & existing_edges) / min(
            len(candidate_edges),
            len(existing_edges),
        )
        if similarity >= threshold:
            return True

    return False


def route_edge_set(route):
    """Return normalized undirected edge keys for a route."""
    return {edge_key(u, v) for u, v in route_edges(route)}


def route_weight(graph, route):
    """Return weighted route length, ignoring non-edges defensively."""
    total = 0.0

    for u, v in route_edges(route):
        if graph.has_edge(u, v):
            total += graph[u][v]["weight"]

    return total


def max_graph_edge_weight(graph):
    """Return the largest graph edge weight."""
    return max((data["weight"] for _, _, data in graph.edges(data=True)), default=0.0)


def route_districts(graph, route):
    """Return district ids touched by a route."""
    return {
        graph.nodes[stop_id].get("district")
        for stop_id in route
        if stop_id in graph.nodes and graph.nodes[stop_id].get("district") is not None
    }


def run_baseline(graph, district_ids, od_matrix, stops_by_district, stop_to_district, config):
    """Build baseline routes and evaluate them with the common metric function."""
    routes = build_baseline_routes(graph, district_ids, od_matrix, stops_by_district, config)
    metrics = evaluate_network(graph, routes, stop_to_district, district_ids, od_matrix, config)
    return routes, metrics
