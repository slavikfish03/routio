"""Graph construction utilities for the Routio experiment."""

from __future__ import annotations

import math
import warnings
from numbers import Real

import networkx as nx
from matplotlib.path import Path


def euclidean_distance(p1, p2):
    """Return Euclidean distance between two 2D points."""
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x2 - x1, y2 - y1)


def point_in_polygon(point, polygon):
    """Return True when a point lies inside a polygon."""
    return Path(polygon).contains_point(point)


def assign_stop_districts(stops, districts):
    """Map every stop id to the first district polygon containing its point."""
    stop_to_district = {}

    for stop_id, point in stops.items():
        district_id = None
        for candidate_id, district in districts.items():
            if point_in_polygon(point, district["polygon"]):
                district_id = candidate_id
                break
        stop_to_district[stop_id] = district_id

    missing = [stop_id for stop_id, district_id in stop_to_district.items() if district_id is None]
    if missing:
        warnings.warn(
            f"{len(missing)} stops are outside all district polygons: {missing}",
            RuntimeWarning,
            stacklevel=2,
        )

    return stop_to_district


def get_stops_by_district(stop_to_district, district_ids):
    """Group stop ids by district id."""
    stops_by_district = {district_id: [] for district_id in district_ids}

    for stop_id, district_id in stop_to_district.items():
        if district_id in stops_by_district:
            stops_by_district[district_id].append(stop_id)

    return stops_by_district


def build_graph(stops, edges, districts):
    """Build an undirected road graph with Euclidean edge weights."""
    stop_to_district = assign_stop_districts(stops, districts)
    graph = nx.Graph()

    for stop_id, point in stops.items():
        x, y = point
        graph.add_node(stop_id, x=x, y=y, pos=(x, y), district=stop_to_district[stop_id])

    for u, v in edges:
        weight = euclidean_distance(stops[u], stops[v])
        graph.add_edge(u, v, weight=weight)

    return graph


def validate_input_data(districts, district_ids, od_matrix, stops, edges, config):
    """Validate experiment inputs before building the graph."""
    errors = []

    if set(district_ids) != set(districts):
        errors.append("district_ids must contain exactly the same ids as districts.")

    if len(od_matrix) != len(district_ids):
        errors.append("od_matrix row count must match len(district_ids).")
    else:
        for row_idx, row in enumerate(od_matrix):
            if len(row) != len(district_ids):
                errors.append(f"od_matrix row {row_idx} length must match len(district_ids).")
            for col_idx, value in enumerate(row):
                if not isinstance(value, Real):
                    errors.append(f"od_matrix[{row_idx}][{col_idx}] must be numeric.")
                elif not 0 <= value <= 1:
                    errors.append(f"od_matrix[{row_idx}][{col_idx}] must be in [0, 1].")

    if not stops:
        errors.append("stops is empty. Fill stops in data_input.py before running the experiment.")
    else:
        for stop_id, point in stops.items():
            if not isinstance(stop_id, str):
                errors.append(f"stop id {stop_id!r} must be a string.")
            if not _is_point(point):
                errors.append(f"stop {stop_id!r} must have coordinates as a 2-number list or tuple.")

    if not edges:
        errors.append("edges is empty. Fill edges in data_input.py before running the experiment.")
    else:
        for edge_idx, edge in enumerate(edges):
            if not _is_edge(edge):
                errors.append(f"edge #{edge_idx} must be a pair of stop ids.")
                continue
            u, v = edge
            if u not in stops:
                errors.append(f"edge #{edge_idx} references unknown stop {u!r}.")
            if v not in stops:
                errors.append(f"edge #{edge_idx} references unknown stop {v!r}.")

    _validate_config(config, errors)

    if errors:
        message = "Invalid input data:\n" + "\n".join(f"- {error}" for error in errors)
        raise ValueError(message)


def validate_graph(graph):
    """Validate the built graph and warn about non-fatal topology issues."""
    if graph.number_of_nodes() == 0:
        raise ValueError("Graph has no nodes.")
    if graph.number_of_edges() == 0:
        raise ValueError("Graph has no edges.")

    missing_weight_edges = [(u, v) for u, v, data in graph.edges(data=True) if "weight" not in data]
    if missing_weight_edges:
        raise ValueError(f"Graph edges without weight: {missing_weight_edges}")

    non_positive_edges = [
        (u, v, data["weight"])
        for u, v, data in graph.edges(data=True)
        if data["weight"] <= 0
    ]
    if non_positive_edges:
        raise ValueError(f"Graph edges must have positive weight: {non_positive_edges}")

    if not nx.is_connected(graph):
        components = nx.number_connected_components(graph)
        warnings.warn(
            f"Graph is not connected: {components} connected components.",
            RuntimeWarning,
            stacklevel=2,
        )


def edge_key(u, v):
    """Return a stable key for an undirected edge."""
    return tuple(sorted((u, v)))


def route_edges(route):
    """Return consecutive stop pairs from a route."""
    return list(zip(route[:-1], route[1:]))


def _is_point(point):
    return (
        isinstance(point, (list, tuple))
        and len(point) == 2
        and isinstance(point[0], Real)
        and isinstance(point[1], Real)
    )


def _is_edge(edge):
    return (
        isinstance(edge, (list, tuple))
        and len(edge) == 2
        and isinstance(edge[0], str)
        and isinstance(edge[1], str)
    )


def _validate_config(config, errors):
    required_keys = [
        "num_routes",
        "min_route_length",
        "max_route_length",
        "population_size",
        "elite_size",
        "alpha_demand_mismatch",
        "beta_uncovered_demand",
        "gamma_length_penalty",
        "delta_overlap_penalty",
    ]
    for key in required_keys:
        if key not in config:
            errors.append(f"config is missing {key!r}.")

    if config.get("num_routes", 0) <= 0:
        errors.append("config['num_routes'] must be > 0.")
    if config.get("min_route_length", 0) < 2:
        errors.append("config['min_route_length'] must be >= 2.")
    if config.get("max_route_length", 0) < config.get("min_route_length", 0):
        errors.append("config['max_route_length'] must be >= config['min_route_length'].")
    if config.get("population_size", 0) < config.get("elite_size", 0):
        errors.append("config['population_size'] must be >= config['elite_size'].")

    loss_weight_keys = [
        "alpha_demand_mismatch",
        "beta_uncovered_demand",
        "gamma_length_penalty",
        "delta_overlap_penalty",
    ]
    for key in loss_weight_keys:
        value = config.get(key)
        if value is not None and value < 0:
            errors.append(f"config[{key!r}] must be non-negative.")
