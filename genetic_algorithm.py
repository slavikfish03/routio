"""Genetic algorithm implementation for the Routio experiment."""

from __future__ import annotations

import random

import networkx as nx

from metrics import evaluate_network


def random_walk_route(graph, config, rng):
    """Generate one route as a random walk without repeated stops."""
    nodes = list(graph.nodes)
    if not nodes:
        raise ValueError("Cannot generate a route from an empty graph.")

    min_length = config["min_route_length"]
    max_length = config["max_route_length"]
    best_route = None

    for _ in range(30):
        target_length = rng.randint(min_length, max_length)
        route = [rng.choice(nodes)]
        _grow_route(graph, route, target_length, rng, min_length)
        _extend_route_to_min_length(graph, route, config, rng)
        if len(route) >= min_length:
            return route
        if best_route is None or len(route) > len(best_route):
            best_route = list(route)

    if best_route and len(best_route) >= 2:
        return best_route

    return _fallback_route(graph, nodes, rng)


def random_individual(graph, config, rng):
    """Generate one individual: a full route network."""
    return [random_walk_route(graph, config, rng) for _ in range(config["num_routes"])]


def initial_population(graph, config, rng):
    """Generate the initial GA population."""
    return [random_individual(graph, config, rng) for _ in range(config["population_size"])]


def score_population(population, graph, stop_to_district, district_ids, od_matrix, config):
    """Evaluate every individual with the common network metrics."""
    scored = []

    for individual in population:
        metrics = evaluate_network(
            graph,
            individual,
            stop_to_district,
            district_ids,
            od_matrix,
            config,
        )
        scored.append(
            {
                "individual": _copy_individual(individual),
                "metrics": metrics,
                "loss": metrics["loss"],
            }
        )

    return scored


def tournament_selection(scored_population, config, rng):
    """Select one parent via tournament selection by minimum loss."""
    tournament_size = min(config["tournament_size"], len(scored_population))
    candidates = rng.sample(scored_population, tournament_size)
    winner = min(candidates, key=lambda item: item["loss"])
    return _copy_individual(winner["individual"])


def crossover(parent_a, parent_b, config, rng):
    """Perform simple route-level crossover between two route networks."""
    if rng.random() > config["crossover_rate"]:
        return _copy_individual(parent_a if rng.random() < 0.5 else parent_b)

    route_count = min(len(parent_a), len(parent_b))
    if route_count < 2:
        return _copy_individual(parent_a)

    split = rng.randint(1, route_count - 1)
    child = _copy_individual(parent_a[:split]) + _copy_individual(parent_b[split:route_count])

    while len(child) < config["num_routes"]:
        child.append(list(rng.choice(parent_a if rng.random() < 0.5 else parent_b)))

    return child[: config["num_routes"]]


def mutate_route(graph, route, config, rng):
    """Mutate one route, then repair it for graph validity."""
    operation = rng.choice(["replace", "reroute_tail", "delete_stop", "insert_neighbor", "close_cycle"])
    route = list(route)

    if operation == "replace":
        route = random_walk_route(graph, config, rng)
    elif operation == "reroute_tail" and route:
        cut_idx = rng.randrange(len(route))
        route = route[: cut_idx + 1]
        target_length = rng.randint(config["min_route_length"], config["max_route_length"])
        _grow_route(graph, route, target_length, rng, config["min_route_length"])
    elif operation == "delete_stop" and len(route) > config["min_route_length"]:
        del route[rng.randrange(len(route))]
    elif operation == "insert_neighbor" and len(route) < config["max_route_length"] and route:
        insert_after = rng.randrange(len(route))
        visited = _internal_visited(route)
        neighbors = [node for node in graph.neighbors(route[insert_after]) if node not in visited]
        if neighbors:
            route.insert(insert_after + 1, rng.choice(neighbors))
    elif operation == "close_cycle":
        _close_route_if_possible(graph, route, config)

    return repair_route(graph, route, config, rng)


def mutate_individual(graph, individual, config, rng):
    """Mutate routes in a full route network."""
    mutated = []

    for route in individual:
        if rng.random() < config["mutation_rate"]:
            mutated.append(mutate_route(graph, route, config, rng))
        else:
            mutated.append(repair_route(graph, route, config, rng))

    while len(mutated) < config["num_routes"]:
        mutated.append(random_walk_route(graph, config, rng))

    return mutated[: config["num_routes"]]


def repair_route(graph, route, config, rng):
    """Repair structural validity and keep routes as simple paths."""
    nodes = set(graph.nodes)
    route = [stop_id for stop_id in route if stop_id in nodes]

    if not route:
        return random_walk_route(graph, config, rng)

    repaired = [route[0]]
    visited = {route[0]}
    max_length = config["max_route_length"]
    min_length = config["min_route_length"]

    for idx, stop_id in enumerate(route[1:], start=1):
        current = repaired[-1]
        is_final_closure = idx == len(route) - 1 and stop_id == repaired[0]

        if is_final_closure:
            if (
                len(repaired) >= 3
                and len(repaired) < max_length
                and graph.has_edge(current, stop_id)
            ):
                repaired.append(stop_id)
            break

        if current == stop_id:
            continue
        if stop_id in visited:
            continue
        if graph.has_edge(current, stop_id):
            repaired.append(stop_id)
            visited.add(stop_id)
            if len(repaired) >= max_length:
                break
            continue

        try:
            path = nx.shortest_path(graph, current, stop_id, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        if any(node in visited for node in path[1:]):
            continue
        if len(repaired) + len(path[1:]) > max_length:
            continue

        repaired.extend(path[1:])
        visited.update(path[1:])

    if is_explicit_cycle(repaired) and len(repaired) < min_length:
        repaired = repaired[:-1]
    if len(repaired) < min_length:
        _grow_route(graph, repaired, min_length, rng, min_length)
    if len(repaired) < min_length:
        _extend_route_to_min_length(graph, repaired, config, rng)
    if len(repaired) >= min_length:
        _close_route_if_possible(graph, repaired, config, close_probability=0.35, rng=rng)

    if len(repaired) < 2:
        return random_walk_route(graph, config, rng)

    return repaired[:max_length]


def run_ga(graph, stop_to_district, district_ids, od_matrix, config):
    """Run the genetic algorithm and return best routes, metrics, and history."""
    rng = random.Random(config.get("random_seed", 42))
    population = initial_population(graph, config, rng)
    best_individual = None
    best_metrics = None
    best_loss = None
    history = []

    for generation in range(config["num_generations"]):
        scored = score_population(
            population,
            graph,
            stop_to_district,
            district_ids,
            od_matrix,
            config,
        )
        scored.sort(key=lambda item: item["loss"])

        if best_loss is None or scored[0]["loss"] < best_loss:
            best_loss = scored[0]["loss"]
            best_individual = _copy_individual(scored[0]["individual"])
            best_metrics = dict(scored[0]["metrics"])

        mean_loss = sum(item["loss"] for item in scored) / len(scored)
        history.append(_history_row(generation, scored[0]["metrics"], mean_loss))

        next_population = [
            _copy_individual(item["individual"])
            for item in scored[: config["elite_size"]]
        ]

        while len(next_population) < config["population_size"]:
            parent_a = tournament_selection(scored, config, rng)
            parent_b = tournament_selection(scored, config, rng)
            child = crossover(parent_a, parent_b, config, rng)
            child = mutate_individual(graph, child, config, rng)
            next_population.append(child)

        population = next_population

    final_scored = score_population(
        population,
        graph,
        stop_to_district,
        district_ids,
        od_matrix,
        config,
    )
    final_best = min(final_scored, key=lambda item: item["loss"])
    if best_loss is None or final_best["loss"] < best_loss:
        best_individual = _copy_individual(final_best["individual"])
        best_metrics = dict(final_best["metrics"])

    return best_individual, best_metrics, history


def _grow_route(graph, route, target_length, rng, min_length=2):
    while len(route) < target_length:
        if (
            len(route) + 1 >= min_length
            and len(route) < target_length
            and _can_close_route(graph, route)
            and rng.random() < 0.2
        ):
            route.append(route[0])
            break

        visited = _internal_visited(route)
        neighbors = [node for node in graph.neighbors(route[-1]) if node not in visited]
        if not neighbors:
            break
        route.append(rng.choice(neighbors))

    if (
        len(route) + 1 >= min_length
        and len(route) < target_length
        and _can_close_route(graph, route)
        and rng.random() < 0.5
    ):
        route.append(route[0])


def _extend_route_to_min_length(graph, route, config, rng):
    """Extend a short simple route from either end when one-way growth is stuck."""
    min_length = config["min_route_length"]
    max_length = config["max_route_length"]

    while len(route) < min_length and len(route) < max_length:
        candidates = _extension_candidates(graph, route)
        if not candidates:
            break
        route[:] = rng.choice(candidates)


def _extension_candidates(graph, route):
    if not route or is_explicit_cycle(route):
        return []

    visited = set(route)
    candidates = []

    for neighbor in graph.neighbors(route[0]):
        if neighbor not in visited:
            candidates.append([neighbor, *route])

    for neighbor in graph.neighbors(route[-1]):
        if neighbor not in visited:
            candidates.append([*route, neighbor])

    return candidates


def _fallback_route(graph, nodes, rng):
    start = rng.choice(nodes)
    neighbors = list(graph.neighbors(start))
    if not neighbors:
        return [start]
    return [start, rng.choice(neighbors)]


def is_simple_route(route):
    """Return True when a route has no repeats except optional first/last closure."""
    if is_explicit_cycle(route):
        return len(route[:-1]) == len(set(route[:-1]))
    return len(route) == len(set(route))


def is_explicit_cycle(route):
    """Return True when a route explicitly starts and ends at the same stop."""
    return len(route) >= 4 and route[0] == route[-1]


def _internal_visited(route):
    if is_explicit_cycle(route):
        return set(route[:-1])
    return set(route)


def _can_close_route(graph, route):
    return (
        len(route) >= 3
        and not is_explicit_cycle(route)
        and route[0] != route[-1]
        and graph.has_edge(route[-1], route[0])
    )


def _close_route_if_possible(graph, route, config, close_probability=1.0, rng=None):
    if len(route) >= config["max_route_length"]:
        return
    if not _can_close_route(graph, route):
        return
    if rng is not None and rng.random() > close_probability:
        return
    route.append(route[0])


def _copy_individual(individual):
    return [list(route) for route in individual]


def _history_row(generation, metrics, mean_loss):
    return {
        "generation": generation,
        "best_loss": metrics["loss"],
        "mean_loss": mean_loss,
        "demand_mismatch": metrics["demand_mismatch"],
        "uncovered_demand_penalty": metrics["uncovered_demand_penalty"],
        "length_penalty": metrics["length_penalty"],
        "overlap_penalty": metrics["overlap_penalty"],
    }
