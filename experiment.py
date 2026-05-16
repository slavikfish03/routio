"""Experiment orchestration for the Routio project."""

from __future__ import annotations

import csv
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


METRIC_COLUMNS = [
    "loss",
    "demand_mismatch",
    "uncovered_demand_penalty",
    "length_penalty",
    "overlap_penalty",
    "total_length",
    "covered_demand_share",
    "num_potentially_circular_routes",
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


def run_experiment():
    """Run the full baseline-vs-GA experiment and save all artifacts."""
    try:
        validate_input_data(districts, district_ids, od_matrix, stops, edges, config)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    output_dir = Path(config.get("output_dir", "outputs"))
    route_colors = config["route_colors"]
    ensure_output_dirs(output_dir)

    graph = build_graph(stops, edges, districts)
    validate_graph(graph)

    stop_to_district = {node: graph.nodes[node]["district"] for node in graph.nodes}
    stops_by_district = get_stops_by_district(stop_to_district, district_ids)

    plot_city_graph(graph, districts, stops, output_dir / "01_city_graph.png", config)
    plot_od_heatmap(od_matrix, district_ids, output_dir / "02_od_matrix_heatmap.png", config)

    baseline_routes, baseline_metrics = run_baseline(
        graph,
        district_ids,
        od_matrix,
        stops_by_district,
        stop_to_district,
        config,
    )
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        baseline_routes,
        route_colors,
        "Baseline routes",
        output_dir / "03_baseline_routes_colored.png",
        config,
    )
    plot_edge_load_map(
        graph,
        districts,
        stops,
        baseline_routes,
        "Baseline edge load",
        output_dir / "05_baseline_edge_load.png",
        config,
    )

    ga_routes, ga_metrics, ga_history = run_ga(
        graph,
        stop_to_district,
        district_ids,
        od_matrix,
        config,
    )
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        ga_routes,
        route_colors,
        "GA routes",
        output_dir / "04_ga_routes_colored.png",
        config,
    )
    plot_edge_load_map(
        graph,
        districts,
        stops,
        ga_routes,
        "GA edge load",
        output_dir / "06_ga_edge_load.png",
        config,
    )

    xlim, ylim = get_center_bounds(districts)
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        ga_routes,
        route_colors,
        "GA routes: center zoom",
        output_dir / "07_ga_center_zoom_colored.png",
        config,
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
        config,
        xlim=xlim,
        ylim=ylim,
    )

    plot_loss_history(ga_history, output_dir / "09_ga_loss_history.png", config)
    plot_components_history(ga_history, output_dir / "10_ga_components_history.png", config)
    plot_metrics_comparison(
        baseline_metrics,
        ga_metrics,
        output_dir / "11_method_comparison.png",
        config,
    )

    plot_individual_routes(
        graph,
        districts,
        stops,
        baseline_routes,
        route_colors,
        output_dir,
        "baseline",
        config,
    )
    plot_individual_routes(
        graph,
        districts,
        stops,
        ga_routes,
        route_colors,
        output_dir,
        "ga",
        config,
    )

    save_metrics_csv(baseline_metrics, ga_metrics, output_dir)
    save_ga_history_csv(ga_history, output_dir)
    print_summary(baseline_metrics, ga_metrics, baseline_routes, ga_routes, output_dir)

    return {
        "baseline_routes": baseline_routes,
        "baseline_metrics": baseline_metrics,
        "ga_routes": ga_routes,
        "ga_metrics": ga_metrics,
        "ga_history": ga_history,
    }


def save_metrics_csv(baseline_metrics, ga_metrics, output_dir):
    """Save baseline-vs-GA metrics as a CSV file."""
    output_path = Path(output_dir) / "metrics_comparison.csv"
    rows = [
        _metrics_row("baseline", baseline_metrics),
        _metrics_row("ga", ga_metrics),
    ]

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["method", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def save_ga_history_csv(ga_history, output_dir):
    """Save GA history as a CSV file."""
    output_path = Path(output_dir) / "ga_history.csv"

    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=GA_HISTORY_COLUMNS)
        writer.writeheader()
        for row in ga_history:
            writer.writerow({key: row.get(key, "") for key in GA_HISTORY_COLUMNS})


def print_summary(baseline_metrics, ga_metrics, baseline_routes, ga_routes, output_dir):
    """Print a concise experiment summary."""
    print("Experiment completed.")
    print(f"Outputs: {Path(output_dir).resolve()}")
    print(f"Baseline routes: {len(baseline_routes)}")
    print(f"GA routes: {len(ga_routes)}")
    print("")
    print("Metric comparison:")
    for key in METRIC_COLUMNS:
        baseline_value = baseline_metrics.get(key, 0.0)
        ga_value = ga_metrics.get(key, 0.0)
        print(f"  {key}: baseline={baseline_value:.4f}, ga={ga_value:.4f}")


def _metrics_row(method, metrics):
    row = {"method": method}
    for key in METRIC_COLUMNS:
        row[key] = metrics.get(key, "")
    return row
