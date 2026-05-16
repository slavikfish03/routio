"""Visualization helpers for the Routio experiment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from graph_utils import edge_key, route_edges
from metrics import edge_usage_counts


def ensure_output_dirs(output_dir):
    """Create the output directory and the route subdirectory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "routes").mkdir(parents=True, exist_ok=True)


def save_figure(fig, save_path, config=None):
    """Save a matplotlib figure as PNG and close it."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    dpi = 200 if config is None else config.get("figure_dpi", 200)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_city_graph(graph, districts, stops, save_path, config):
    """Plot district polygons, stops, and the base road graph."""
    fig, ax = plt.subplots(figsize=(10, 9))
    _draw_districts(ax, districts, config, label=True)
    _draw_base_graph(ax, graph, stops, config)
    _draw_stops(ax, stops)
    _finish_map(ax, "Synthetic city: districts, stops, and roads")
    save_figure(fig, save_path, config)


def plot_od_heatmap(od_matrix, district_ids, save_path, config):
    """Plot the OD matrix as a heatmap."""
    values = np.array(od_matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(values, cmap="YlOrRd", vmin=0, vmax=max(1.0, values.max()))
    ax.set_xticks(range(len(district_ids)))
    ax.set_yticks(range(len(district_ids)))
    ax.set_xticklabels(district_ids, rotation=35, ha="right")
    ax.set_yticklabels(district_ids)
    ax.set_xlabel("To district")
    ax.set_ylabel("From district")
    ax.set_title("OD demand matrix")

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Demand")
    save_figure(fig, save_path, config)


def build_edge_route_index(routes):
    """Return mapping edge_key -> list of route indices using that edge."""
    edge_to_routes = {}

    for route_idx, route in enumerate(routes):
        for u, v in route_edges(route):
            key = edge_key(u, v)
            route_indices = edge_to_routes.setdefault(key, [])
            if route_idx not in route_indices:
                route_indices.append(route_idx)

    return edge_to_routes


def get_offset_segment(u, v, stops, route_position, routes_count, offset_step):
    """Return shifted segment coordinates for parallel route drawing."""
    x1, y1 = stops[u]
    x2, y2 = stops[v]
    dx = x2 - x1
    dy = y2 - y1
    length = float(np.hypot(dx, dy))

    if length == 0 or routes_count <= 1:
        return x1, y1, x2, y2

    nx = -dy / length
    ny = dx / length
    offset = (route_position - (routes_count - 1) / 2) * offset_step

    return (
        x1 + nx * offset,
        y1 + ny * offset,
        x2 + nx * offset,
        y2 + ny * offset,
    )


def plot_routes_with_offsets(
    graph,
    districts,
    stops,
    routes,
    route_colors,
    title,
    save_path,
    config,
    xlim=None,
    ylim=None,
):
    """Plot all routes with colors, white underlays, and offsets on shared edges."""
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_districts(ax, districts, config, label=True)
    _draw_base_graph(ax, graph, stops, config)

    edge_to_routes = build_edge_route_index(routes)
    route_width = config.get("route_line_width", 2.8)
    underlay_width = config.get("route_underlay_width", route_width + 2.4)
    offset_step = config.get("route_offset_step", 1.15)

    legend_handles = []
    for route_idx, route in enumerate(routes):
        color = route_colors[route_idx % len(route_colors)]
        legend_handles.append(Line2D([0], [0], color=color, lw=route_width, label=f"Route {route_idx + 1}"))

        for u, v in route_edges(route):
            if u not in stops or v not in stops:
                continue

            key = edge_key(u, v)
            route_indices = edge_to_routes.get(key, [route_idx])
            route_position = route_indices.index(route_idx) if route_idx in route_indices else 0
            routes_count = len(route_indices)
            x1, y1, x2, y2 = get_offset_segment(
                u,
                v,
                stops,
                route_position,
                routes_count,
                offset_step,
            )

            ax.plot(
                [x1, x2],
                [y1, y2],
                color="white",
                linewidth=underlay_width,
                alpha=0.9,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=4,
            )
            ax.plot(
                [x1, x2],
                [y1, y2],
                color=color,
                linewidth=route_width,
                alpha=0.95,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=5,
            )

    _draw_stops(ax, stops)
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8, frameon=True)
    _finish_map(ax, title, xlim=xlim, ylim=ylim)
    save_figure(fig, save_path, config)


def plot_edge_load_map(graph, districts, stops, routes, title, save_path, config):
    """Plot edge load where line width is based on route edge usage count."""
    fig, ax = plt.subplots(figsize=(11, 9))
    _draw_districts(ax, districts, config, label=True)
    _draw_base_graph(ax, graph, stops, config)

    usage = edge_usage_counts(routes)
    max_load = max(usage.values(), default=0)

    for (u, v), count in usage.items():
        if u not in stops or v not in stops:
            continue
        x1, y1 = stops[u]
        x2, y2 = stops[v]
        linewidth = 1.4 + 1.4 * count
        alpha = 0.45 + 0.45 * (count / max_load) if max_load else 0.45
        ax.plot(
            [x1, x2],
            [y1, y2],
            color="#b2182b",
            linewidth=linewidth,
            alpha=alpha,
            solid_capstyle="round",
            zorder=4,
        )

    _draw_stops(ax, stops)
    if max_load:
        handles = [
            Line2D([0], [0], color="#b2182b", lw=1.4 + 1.4 * load, label=f"{load} route(s)")
            for load in range(1, max_load + 1)
        ]
        ax.legend(handles=handles, title="Edge load", loc="upper left", fontsize=8, frameon=True)
    _finish_map(ax, title)
    save_figure(fig, save_path, config)


def plot_loss_history(ga_history, save_path, config):
    """Plot best and mean loss by generation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    if ga_history:
        generations = _history_values(ga_history, "generation")
        ax.plot(generations, _history_values(ga_history, "best_loss"), label="Best loss", linewidth=2.2)
        ax.plot(generations, _history_values(ga_history, "mean_loss"), label="Mean loss", linewidth=1.8)
        ax.legend()
    else:
        _draw_no_data(ax)
    ax.set_title("GA loss history")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    save_figure(fig, save_path, config)


def plot_components_history(ga_history, save_path, config):
    """Plot objective components for the best solution by generation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    component_keys = [
        "demand_mismatch",
        "uncovered_demand_penalty",
        "length_penalty",
        "overlap_penalty",
    ]

    if ga_history:
        generations = _history_values(ga_history, "generation")
        for key in component_keys:
            ax.plot(generations, _history_values(ga_history, key), label=key, linewidth=1.8)
        ax.legend(fontsize=8)
    else:
        _draw_no_data(ax)

    ax.set_title("GA loss components history")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Component value")
    ax.grid(alpha=0.25)
    save_figure(fig, save_path, config)


def plot_metrics_comparison(baseline_metrics, ga_metrics, save_path, config):
    """Plot a baseline-vs-GA comparison for the main objective components."""
    metric_keys = [
        "loss",
        "demand_mismatch",
        "uncovered_demand_penalty",
        "length_penalty",
        "overlap_penalty",
    ]
    labels = ["loss", "demand", "uncovered", "length", "overlap"]
    x = np.arange(len(metric_keys))
    width = 0.36

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width / 2, [baseline_metrics.get(key, 0.0) for key in metric_keys], width, label="Baseline")
    ax.bar(x + width / 2, [ga_metrics.get(key, 0.0) for key in metric_keys], width, label="GA")
    ax.set_title("Baseline vs GA")
    ax.set_ylabel("Value")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save_figure(fig, save_path, config)


def plot_individual_routes(graph, districts, stops, routes, route_colors, output_dir, prefix, config):
    """Save a separate colored map for each route."""
    routes_dir = Path(output_dir) / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    for route_idx, route in enumerate(routes):
        save_path = routes_dir / f"{prefix}_route_{route_idx + 1:02d}.png"
        plot_routes_with_offsets(
            graph,
            districts,
            stops,
            [route],
            [route_colors[route_idx % len(route_colors)]],
            f"{prefix.replace('_', ' ').title()} route {route_idx + 1}",
            save_path,
            config,
        )


def get_center_bounds(districts, padding=10):
    """Return xlim and ylim around the district named 'Центр'."""
    center = districts.get("Центр")
    if center is None:
        all_points = [point for district in districts.values() for point in district["polygon"]]
    else:
        all_points = center["polygon"]

    xs = [point[0] for point in all_points]
    ys = [point[1] for point in all_points]
    return (
        (min(xs) - padding, max(xs) + padding),
        (min(ys) - padding, max(ys) + padding),
    )


def _draw_districts(ax, districts, config, label=False):
    alpha = config.get("district_alpha", 0.22)
    for district_id, district in districts.items():
        polygon = np.array(district["polygon"], dtype=float)
        patch = Polygon(
            polygon,
            closed=True,
            facecolor=district.get("color", "#cccccc"),
            edgecolor="#555555",
            linewidth=0.8,
            alpha=alpha,
            zorder=1,
        )
        ax.add_patch(patch)

        if label:
            centroid = polygon.mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                district_id,
                ha="center",
                va="center",
                fontsize=8,
                color="#222222",
                zorder=2,
            )


def _draw_base_graph(ax, graph, stops, config):
    color = "#b0b0b0"
    linewidth = config.get("base_graph_line_width", 0.8)
    alpha = config.get("base_graph_alpha", 0.35)

    for u, v in graph.edges:
        if u not in stops or v not in stops:
            continue
        x1, y1 = stops[u]
        x2, y2 = stops[v]
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )


def _draw_stops(ax, stops):
    if not stops:
        return

    xs = [point[0] for point in stops.values()]
    ys = [point[1] for point in stops.values()]
    ax.scatter(
        xs,
        ys,
        s=24,
        facecolor="white",
        edgecolor="#222222",
        linewidth=0.7,
        zorder=6,
    )


def _finish_map(ax, title, xlim=None, ylim=None):
    ax.set_title(title)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.12)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


def _history_values(ga_history, key):
    return [row.get(key, 0.0) for row in ga_history]


def _draw_no_data(ax):
    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
