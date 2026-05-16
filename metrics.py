"""Metric and loss utilities for the Routio experiment."""

from __future__ import annotations

from graph_utils import edge_key, route_edges


def route_length(graph, route):
    """Return the weighted length of one route."""
    total = 0.0

    for u, v in route_edges(route):
        if not graph.has_edge(u, v):
            raise ValueError(f"Route uses missing graph edge: {u!r} - {v!r}")
        total += graph[u][v]["weight"]

    return total


def network_length(graph, routes):
    """Return total weighted length of all routes."""
    return sum(route_length(graph, route) for route in routes)


def district_target_intensity(od_matrix, district_ids):
    """Return normalized target district importance from incoming and outgoing OD demand."""
    intensities = {}

    for idx, district_id in enumerate(district_ids):
        outgoing = sum(od_matrix[idx])
        incoming = sum(row[idx] for row in od_matrix)
        internal = od_matrix[idx][idx]
        intensities[district_id] = outgoing + incoming - internal

    total = sum(intensities.values())
    if total == 0:
        return {district_id: 0.0 for district_id in district_ids}

    return {district_id: value / total for district_id, value in intensities.items()}


def district_actual_coverage(routes, stop_to_district, district_ids):
    """Return normalized district coverage by stop visits in the route network."""
    coverage = {district_id: 0 for district_id in district_ids}

    for route in routes:
        for stop_id in route:
            district_id = stop_to_district.get(stop_id)
            if district_id in coverage:
                coverage[district_id] += 1

    total = sum(coverage.values())
    if total == 0:
        return {district_id: 0.0 for district_id in district_ids}

    return {district_id: value / total for district_id, value in coverage.items()}


def demand_mismatch(routes, stop_to_district, district_ids, od_matrix):
    """Return L1 mismatch between target demand importance and actual route coverage."""
    target = district_target_intensity(od_matrix, district_ids)
    actual = district_actual_coverage(routes, stop_to_district, district_ids)

    return sum(abs(target[district_id] - actual[district_id]) for district_id in district_ids) / 2


def covered_od_demand(routes, stop_to_district, district_ids, od_matrix):
    """Return covered and total interdistrict OD demand."""
    route_district_sets = []
    for route in routes:
        route_district_sets.append(
            {
                stop_to_district.get(stop_id)
                for stop_id in route
                if stop_to_district.get(stop_id) in district_ids
            }
        )

    covered = 0.0
    total = 0.0

    for from_idx, from_district in enumerate(district_ids):
        for to_idx, to_district in enumerate(district_ids):
            if from_idx == to_idx:
                continue

            demand = od_matrix[from_idx][to_idx]
            total += demand

            if any(
                from_district in route_districts and to_district in route_districts
                for route_districts in route_district_sets
            ):
                covered += demand

    return covered, total


def uncovered_demand_penalty(routes, stop_to_district, district_ids, od_matrix):
    """Return the share of interdistrict OD demand not covered by any single route."""
    covered, total = covered_od_demand(routes, stop_to_district, district_ids, od_matrix)
    if total == 0:
        return 0.0

    return 1 - covered / total


def edge_usage_counts(routes):
    """Return edge usage counts across all route edge occurrences."""
    counts = {}

    for route in routes:
        for u, v in route_edges(route):
            key = edge_key(u, v)
            counts[key] = counts.get(key, 0) + 1

    return counts


def overlap_penalty(routes):
    """Return penalty for repeated use of the same undirected edge."""
    counts = edge_usage_counts(routes)
    total_route_edges = sum(max(0, len(route) - 1) for route in routes)

    if total_route_edges == 0:
        return 1.0

    repeated_edges = sum(max(0, count - 1) for count in counts.values())
    return repeated_edges / total_route_edges


def length_penalty(graph, routes, config):
    """Return normalized total route network length."""
    edge_weights = [data["weight"] for _, _, data in graph.edges(data=True)]
    if not edge_weights:
        return 1.0

    max_edge_weight = max(edge_weights)
    max_possible_total_length = (
        config["num_routes"] * config["max_route_length"] * max_edge_weight
    )
    if max_possible_total_length <= 0:
        return 1.0

    return min(network_length(graph, routes) / max_possible_total_length, 1.0)


def potentially_circular_routes(graph, routes):
    """Return routes that are explicitly or potentially circular."""
    circular = []

    for route in routes:
        is_explicit_cycle = (
            len(route) >= 4
            and route[0] == route[-1]
            and graph.has_edge(route[-2], route[0])
        )
        is_potential_cycle = (
            len(route) >= 3
            and route[0] != route[-1]
            and graph.has_edge(route[-1], route[0])
        )
        if is_explicit_cycle or is_potential_cycle:
            circular.append(route)

    return circular


def evaluate_network(graph, routes, stop_to_district, district_ids, od_matrix, config):
    """Return all objective components and the final weighted loss."""
    mismatch = demand_mismatch(routes, stop_to_district, district_ids, od_matrix)
    uncovered = uncovered_demand_penalty(routes, stop_to_district, district_ids, od_matrix)
    length = length_penalty(graph, routes, config)
    overlap = overlap_penalty(routes)

    loss = (
        config["alpha_demand_mismatch"] * mismatch
        + config["beta_uncovered_demand"] * uncovered
        + config["gamma_length_penalty"] * length
        + config["delta_overlap_penalty"] * overlap
    )

    covered, total = covered_od_demand(routes, stop_to_district, district_ids, od_matrix)
    covered_demand_share = covered / total if total else 0.0
    circular_routes = potentially_circular_routes(graph, routes)

    return {
        "loss": loss,
        "demand_mismatch": mismatch,
        "uncovered_demand_penalty": uncovered,
        "length_penalty": length,
        "overlap_penalty": overlap,
        "total_length": network_length(graph, routes),
        "covered_demand_share": covered_demand_share,
        "num_potentially_circular_routes": len(circular_routes),
    }
