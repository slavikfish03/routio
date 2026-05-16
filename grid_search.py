"""Controlled grid search for the Routio GA experiment."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from itertools import product
from pathlib import Path

from baseline import run_baseline
from data_input import config, district_ids, districts, edges, od_matrix, stops
from genetic_algorithm import run_ga
from graph_utils import (
    build_graph,
    get_stops_by_district,
    validate_graph,
    validate_input_data,
)
from visualization import (
    ensure_output_dirs,
    get_center_bounds,
    plot_city_graph,
    plot_components_history,
    plot_edge_load_map,
    plot_individual_routes,
    plot_loss_history,
    plot_metrics_comparison,
    plot_od_heatmap,
    plot_routes_with_offsets,
)


NUM_ROUTES_VALUES = [8, 10, 12]
MAX_ROUTE_LENGTH_VALUES = [12, 15, 18]
RANDOM_SEEDS = [42, 43, 44]

LOSS_PROFILES = {
    "balanced": {
        "alpha_demand_mismatch": 0.35,
        "beta_uncovered_demand": 0.35,
        "gamma_length_penalty": 0.20,
        "delta_overlap_penalty": 0.10,
    },
    "demand_priority": {
        "alpha_demand_mismatch": 0.40,
        "beta_uncovered_demand": 0.40,
        "gamma_length_penalty": 0.15,
        "delta_overlap_penalty": 0.05,
    },
    "length_priority": {
        "alpha_demand_mismatch": 0.30,
        "beta_uncovered_demand": 0.30,
        "gamma_length_penalty": 0.30,
        "delta_overlap_penalty": 0.10,
    },
    "overlap_priority": {
        "alpha_demand_mismatch": 0.30,
        "beta_uncovered_demand": 0.30,
        "gamma_length_penalty": 0.20,
        "delta_overlap_penalty": 0.20,
    },
}

GRID_RESULT_COLUMNS = [
    "run_id",
    "seed",
    "num_routes",
    "min_route_length",
    "max_route_length",
    "loss_profile",
    "final_loss",
    "demand_mismatch",
    "uncovered_demand_penalty",
    "length_penalty",
    "overlap_penalty",
    "total_length",
    "covered_demand_ratio",
    "number_of_circular_routes",
    "stop_coverage",
    "district_coverage",
]

GA_HISTORY_COLUMNS = [
    "generation",
    "best_loss",
    "mean_loss",
    "demand_mismatch",
    "uncovered_demand_penalty",
    "length_penalty",
    "overlap_penalty",
]


def run_grid_search(
    num_routes_values=None,
    max_route_length_values=None,
    loss_profiles=None,
    random_seeds=None,
    base_config=None,
    output_dir=None,
    render_best=True,
):
    """Run the controlled grid search and save tabular results."""
    base_config = deepcopy(config if base_config is None else base_config)
    output_dir = Path(output_dir or base_config.get("output_dir", "outputs"))
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        validate_input_data(districts, district_ids, od_matrix, stops, edges, base_config)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    graph = build_graph(stops, edges, districts)
    validate_graph(graph)
    stop_to_district = {node: graph.nodes[node]["district"] for node in graph.nodes}
    stops_by_district = get_stops_by_district(stop_to_district, district_ids)

    num_routes_values = num_routes_values or NUM_ROUTES_VALUES
    max_route_length_values = max_route_length_values or MAX_ROUTE_LENGTH_VALUES
    loss_profiles = loss_profiles or LOSS_PROFILES
    random_seeds = random_seeds or RANDOM_SEEDS

    rows = []
    best_run = None
    run_id = 1
    total_runs = (
        len(num_routes_values)
        * len(max_route_length_values)
        * len(loss_profiles)
        * len(random_seeds)
    )

    for num_routes, max_route_length, profile_name, seed in product(
        num_routes_values,
        max_route_length_values,
        loss_profiles,
        random_seeds,
    ):
        run_config = build_run_config(
            base_config,
            num_routes,
            max_route_length,
            profile_name,
            loss_profiles[profile_name],
            seed,
        )
        print(
            f"[{run_id}/{total_runs}] "
            f"routes={num_routes}, max_len={max_route_length}, "
            f"profile={profile_name}, seed={seed}"
        )

        routes, metrics, history = run_ga(
            graph,
            stop_to_district,
            district_ids,
            od_matrix,
            run_config,
        )
        row = build_result_row(
            run_id,
            seed,
            profile_name,
            run_config,
            routes,
            metrics,
            graph,
            stop_to_district,
        )
        rows.append(row)

        if best_run is None or row["final_loss"] < best_run["row"]["final_loss"]:
            best_run = {
                "row": row,
                "config": deepcopy(run_config),
                "routes": deepcopy(routes),
                "metrics": dict(metrics),
                "history": deepcopy(history),
            }

        run_id += 1

    rows.sort(key=lambda row: row["final_loss"])
    save_grid_results(rows, output_dir)
    save_top_results(rows, output_dir, top_n=10)

    if best_run is not None:
        save_best_config(best_run, output_dir)
        save_best_history(best_run["history"], output_dir)
        if render_best:
            render_best_run(
                graph,
                stops_by_district,
                stop_to_district,
                best_run,
                output_dir / "grid_search_best",
            )
        print_top_results(rows, top_n=10)

    print(f"Grid search completed. Outputs: {output_dir.resolve()}")
    return rows


def build_run_config(base_config, num_routes, max_route_length, profile_name, profile, seed):
    """Return one config variant for a grid search run."""
    run_config = deepcopy(base_config)
    run_config["num_routes"] = num_routes
    run_config["max_route_length"] = max_route_length
    run_config["random_seed"] = seed
    run_config["loss_profile"] = profile_name
    run_config.update(profile)
    return run_config


def build_result_row(
    run_id,
    seed,
    profile_name,
    run_config,
    routes,
    metrics,
    graph,
    stop_to_district,
):
    """Return one CSV-ready result row for a finished GA run."""
    return {
        "run_id": run_id,
        "seed": seed,
        "num_routes": run_config["num_routes"],
        "min_route_length": run_config["min_route_length"],
        "max_route_length": run_config["max_route_length"],
        "loss_profile": profile_name,
        "final_loss": metrics["loss"],
        "demand_mismatch": metrics["demand_mismatch"],
        "uncovered_demand_penalty": metrics["uncovered_demand_penalty"],
        "length_penalty": metrics["length_penalty"],
        "overlap_penalty": metrics["overlap_penalty"],
        "total_length": metrics["total_length"],
        "covered_demand_ratio": metrics["covered_demand_share"],
        "number_of_circular_routes": metrics["num_potentially_circular_routes"],
        "stop_coverage": stop_coverage(routes, graph),
        "district_coverage": district_coverage(routes, stop_to_district),
    }


def stop_coverage(routes, graph):
    """Return share of graph stops used by at least one route."""
    if graph.number_of_nodes() == 0:
        return 0.0
    used_stops = {stop_id for route in routes for stop_id in route}
    return len(used_stops) / graph.number_of_nodes()


def district_coverage(routes, stop_to_district):
    """Return share of districts touched by at least one route."""
    all_districts = {district_id for district_id in stop_to_district.values() if district_id is not None}
    if not all_districts:
        return 0.0

    covered_districts = {
        stop_to_district.get(stop_id)
        for route in routes
        for stop_id in route
        if stop_to_district.get(stop_id) is not None
    }
    return len(covered_districts) / len(all_districts)


def save_grid_results(rows, output_dir):
    """Save all grid search rows sorted by final loss."""
    output_path = Path(output_dir) / "grid_search_results.csv"
    save_rows_csv(rows, output_path)


def save_top_results(rows, output_dir, top_n=10):
    """Save top-N grid search rows sorted by final loss."""
    output_path = Path(output_dir) / "grid_search_top10.csv"
    save_rows_csv(rows[:top_n], output_path)


def save_rows_csv(rows, output_path):
    """Save grid search rows to CSV."""
    with Path(output_path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GRID_RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def save_best_config(best_run, output_dir):
    """Save the best run config as JSON."""
    output_path = Path(output_dir) / "best_config.json"
    payload = {
        "selection_rule": "minimum final_loss",
        "best_run": best_run["row"],
        "config": best_run["config"],
    }
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)


def save_best_history(history, output_dir):
    """Save GA history for the best grid search run."""
    output_path = Path(output_dir) / "grid_search_best_history.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GA_HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key, "") for key in GA_HISTORY_COLUMNS})


def render_best_run(graph, stops_by_district, stop_to_district, best_run, output_dir):
    """Save the full presentation artifact set for the best GA run."""
    best_config = best_run["config"]
    best_routes = best_run["routes"]
    best_metrics = best_run["metrics"]
    route_colors = best_config["route_colors"]

    ensure_output_dirs(output_dir)

    baseline_routes, baseline_metrics = run_baseline(
        graph,
        district_ids,
        od_matrix,
        stops_by_district,
        stop_to_district,
        best_config,
    )

    plot_city_graph(graph, districts, stops, output_dir / "01_city_graph.png", best_config)
    plot_od_heatmap(od_matrix, district_ids, output_dir / "02_od_matrix_heatmap.png", best_config)
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        baseline_routes,
        route_colors,
        "Baseline routes",
        output_dir / "03_baseline_routes_colored.png",
        best_config,
    )
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        best_routes,
        route_colors,
        "Best grid-search GA routes",
        output_dir / "04_ga_routes_colored.png",
        best_config,
    )
    plot_edge_load_map(
        graph,
        districts,
        stops,
        baseline_routes,
        "Baseline edge load",
        output_dir / "05_baseline_edge_load.png",
        best_config,
    )
    plot_edge_load_map(
        graph,
        districts,
        stops,
        best_routes,
        "Best grid-search GA edge load",
        output_dir / "06_ga_edge_load.png",
        best_config,
    )

    xlim, ylim = get_center_bounds(districts)
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        best_routes,
        route_colors,
        "Best grid-search GA routes: center zoom",
        output_dir / "07_ga_center_zoom_colored.png",
        best_config,
        xlim=xlim,
        ylim=ylim,
    )
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        baseline_routes,
        route_colors,
        "Baseline routes: center zoom",
        output_dir / "08_baseline_center_zoom_colored.png",
        best_config,
        xlim=xlim,
        ylim=ylim,
    )
    plot_loss_history(best_run["history"], output_dir / "09_ga_loss_history.png", best_config)
    plot_components_history(best_run["history"], output_dir / "10_ga_components_history.png", best_config)
    plot_metrics_comparison(
        baseline_metrics,
        best_metrics,
        output_dir / "11_method_comparison.png",
        best_config,
    )
    plot_individual_routes(
        graph,
        districts,
        stops,
        baseline_routes,
        route_colors,
        output_dir,
        "baseline",
        best_config,
    )
    plot_individual_routes(
        graph,
        districts,
        stops,
        best_routes,
        route_colors,
        output_dir,
        "ga",
        best_config,
    )


def print_top_results(rows, top_n=10):
    """Print the top-N grid search rows."""
    print("")
    print(f"Top {min(top_n, len(rows))} runs by final_loss:")
    for row in rows[:top_n]:
        print(
            f"  #{row['run_id']}: loss={row['final_loss']:.4f}, "
            f"routes={row['num_routes']}, max_len={row['max_route_length']}, "
            f"profile={row['loss_profile']}, seed={row['seed']}, "
            f"covered={row['covered_demand_ratio']:.4f}"
        )


if __name__ == "__main__":
    run_grid_search()
