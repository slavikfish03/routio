"""Interactive LLM route-network experiment without external API calls."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path

from data_input import config, district_ids, districts, edges, od_matrix, stops
from graph_utils import build_graph, route_edges, validate_graph, validate_input_data
from metrics import evaluate_network
from visualization import (
    ensure_output_dirs,
    get_center_bounds,
    plot_edge_load_map,
    plot_individual_routes,
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


def run_llm_experiment():
    """Create an LLM prompt, read pasted routes, evaluate them, and save artifacts."""
    try:
        validate_input_data(districts, district_ids, od_matrix, stops, edges, config)
    except ValueError as error:
        raise SystemExit(str(error)) from None

    output_dir = Path(config.get("output_dir", "outputs"))
    ensure_output_dirs(output_dir)

    graph = build_graph(stops, edges, districts)
    validate_graph(graph)
    stop_to_district = {node: graph.nodes[node]["district"] for node in graph.nodes}

    prompt = build_llm_prompt(graph, stop_to_district)
    prompt_path = output_dir / "llm_prompt.md"
    save_text(prompt_path, prompt)

    print(f"LLM prompt saved to: {prompt_path.resolve()}")
    print("")
    print("Paste this prompt into an LLM, then paste the LLM answer below.")
    print("Expected answer: JSON route list, for example [[\"A1\", \"G1\"], [\"L3\", \"K3\"]].")
    print("Finish input with a line containing only END.")
    print("")

    answer = read_multiline_answer()
    routes = parse_llm_routes(answer)
    validate_llm_routes(graph, routes, config)

    metrics = evaluate_network(
        graph,
        routes,
        stop_to_district,
        district_ids,
        od_matrix,
        config,
    )

    save_llm_answer(output_dir, answer)
    save_llm_routes(output_dir, routes)
    save_llm_metrics(output_dir, metrics)
    save_comparison_with_existing_metrics(output_dir, metrics)
    render_llm_outputs(graph, routes, output_dir)
    print_summary(routes, metrics, output_dir)

    return {"routes": routes, "metrics": metrics}


def build_llm_prompt(graph, stop_to_district):
    """Build a self-contained prompt for route generation by an external LLM."""
    top_od_pairs = get_top_od_pairs(limit=14)
    stops_payload = {
        stop_id: {
            "x": point[0],
            "y": point[1],
            "district": stop_to_district.get(stop_id),
        }
        for stop_id, point in stops.items()
    }

    prompt = f"""# Task: generate a public-transport route network

You are a heuristic public-transport planner for a synthetic city.

Return only a valid JSON array of routes. Do not include comments.
Each route must be a list of stop ids:

[
  ["STOP_ID_1", "STOP_ID_2", "STOP_ID_3"],
  ["STOP_ID_4", "STOP_ID_5", "STOP_ID_6"]
]

## Goal

Generate a route network that covers high OD demand while keeping routes readable:

- use exactly {config["num_routes"]} routes if possible;
- each route should usually have {config["min_route_length"]} to {config["max_route_length"]} stops;
- every consecutive pair of stops must be connected by a road edge;
- do not repeat stops inside a route;
- the only allowed repeat is an explicit circular route where the last stop equals the first stop;
- avoid excessive overlap between routes;
- include airport access if useful;
- include central-district connectivity if useful;
- prefer understandable routes over tiny two-stop shuttles.

## Loss function used after your answer

Your answer will be evaluated numerically by the same function as baseline and GA:

loss =
  alpha * demand_mismatch
+ beta * uncovered_demand_penalty
+ gamma * length_penalty
+ delta * overlap_penalty

Weights:

{json.dumps(get_loss_weights(), ensure_ascii=False, indent=2)}

## District order for OD matrix

{json.dumps(district_ids, ensure_ascii=False, indent=2)}

## Districts

{json.dumps(districts, ensure_ascii=False, indent=2)}

## OD matrix

Rows are FROM districts, columns are TO districts, in the district order above.

{json.dumps(od_matrix, ensure_ascii=False, indent=2)}

## Highest OD pairs

{json.dumps(top_od_pairs, ensure_ascii=False, indent=2)}

## Stops

Each stop has coordinates and assigned district.

{json.dumps(stops_payload, ensure_ascii=False, indent=2)}

## Road edges

The graph is undirected. You may only use consecutive stop pairs from this list.

{json.dumps([list(edge) for edge in graph.edges], ensure_ascii=False, indent=2)}

## Required output

Return only JSON. No Markdown. No explanation.
"""
    return prompt


def get_loss_weights():
    """Return objective weights from config."""
    return {
        "alpha_demand_mismatch": config["alpha_demand_mismatch"],
        "beta_uncovered_demand": config["beta_uncovered_demand"],
        "gamma_length_penalty": config["gamma_length_penalty"],
        "delta_overlap_penalty": config["delta_overlap_penalty"],
    }


def get_top_od_pairs(limit=14):
    """Return high-demand directed OD pairs for prompt context."""
    pairs = []

    for from_idx, from_district in enumerate(district_ids):
        for to_idx, to_district in enumerate(district_ids):
            if from_idx == to_idx:
                continue
            pairs.append(
                {
                    "from": from_district,
                    "to": to_district,
                    "demand": od_matrix[from_idx][to_idx],
                }
            )

    return sorted(pairs, key=lambda item: item["demand"], reverse=True)[:limit]


def read_multiline_answer():
    """Read pasted LLM output until the END marker."""
    lines = []

    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def parse_llm_routes(answer):
    """Parse routes from raw LLM output."""
    candidates = [answer]
    candidates.extend(extract_code_blocks(answer))
    candidates.extend(extract_bracket_candidates(answer))

    for candidate in candidates:
        parsed = parse_candidate(candidate)
        routes = normalize_routes_payload(parsed)
        if routes is not None:
            return routes

    raise SystemExit(
        "Could not parse LLM routes. Paste a JSON array like "
        '[[\"A1\", \"G1\", \"H1\"], [\"L3\", \"K3\", \"J3\"]].'
    )


def extract_code_blocks(text):
    """Return Markdown fenced-code contents from text."""
    return re.findall(r"```(?:json|python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)


def extract_bracket_candidates(text):
    """Return broad bracket-based parse candidates."""
    candidates = []
    first_square = text.find("[")
    last_square = text.rfind("]")
    if first_square != -1 and last_square > first_square:
        candidates.append(text[first_square : last_square + 1])

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidates.append(text[first_brace : last_brace + 1])

    return candidates


def parse_candidate(candidate):
    """Parse one JSON/Python-literal candidate."""
    candidate = candidate.strip()
    if not candidate:
        return None

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(candidate)
    except (SyntaxError, ValueError):
        return None


def normalize_routes_payload(payload):
    """Normalize common LLM payload shapes to list[list[str]]."""
    if payload is None:
        return None

    if isinstance(payload, dict):
        for key in ("routes", "llm_routes", "route_network", "network"):
            routes = normalize_routes_payload(payload.get(key))
            if routes is not None:
                return routes
        return None

    if not isinstance(payload, list):
        return None

    routes = []
    for item in payload:
        if isinstance(item, dict):
            item = item.get("stops") or item.get("route")
        if not isinstance(item, list) or not all(isinstance(stop_id, str) for stop_id in item):
            return None
        routes.append(item)

    return routes


def validate_llm_routes(graph, routes, config):
    """Validate route structure before metric evaluation."""
    errors = []

    if not routes:
        errors.append("route list is empty.")

    if len(routes) != config["num_routes"]:
        print(
            f"Warning: expected {config['num_routes']} routes, got {len(routes)}. "
            "Metrics will still be computed."
        )

    for route_idx, route in enumerate(routes, start=1):
        if len(route) < 2:
            errors.append(f"route {route_idx} must contain at least two stops.")
            continue
        if len(route) < config["min_route_length"]:
            print(
                f"Warning: route {route_idx} has {len(route)} stops, "
                f"less than min_route_length={config['min_route_length']}."
            )
        if len(route) > config["max_route_length"]:
            print(
                f"Warning: route {route_idx} has {len(route)} stops, "
                f"greater than max_route_length={config['max_route_length']}."
            )

        repeated_error = repeated_stop_error(route)
        if repeated_error:
            errors.append(f"route {route_idx}: {repeated_error}")

        for stop_id in route:
            if stop_id not in graph:
                errors.append(f"route {route_idx} references unknown stop {stop_id!r}.")

        for u, v in route_edges(route):
            if u in graph and v in graph and not graph.has_edge(u, v):
                errors.append(f"route {route_idx} uses missing edge {u!r} - {v!r}.")

    if errors:
        message = "Invalid LLM routes:\n" + "\n".join(f"- {error}" for error in errors)
        raise SystemExit(message)


def repeated_stop_error(route):
    """Return an error string for disallowed repeated stops."""
    if len(route) >= 4 and route[0] == route[-1]:
        inner = route[:-1]
        if len(inner) != len(set(inner)):
            return "explicit cycle has repeated internal stops."
        return None

    if len(route) != len(set(route)):
        return "route has repeated stops; only route[-1] == route[0] is allowed."

    return None


def save_llm_answer(output_dir, answer):
    """Save raw pasted LLM answer."""
    save_text(Path(output_dir) / "llm_answer_raw.txt", answer)


def save_llm_routes(output_dir, routes):
    """Save parsed LLM routes as JSON."""
    output_path = Path(output_dir) / "llm_routes.json"
    with output_path.open("w", encoding="utf-8") as json_file:
        json.dump(routes, json_file, ensure_ascii=False, indent=2)


def save_llm_metrics(output_dir, metrics):
    """Save LLM metrics as CSV."""
    output_path = Path(output_dir) / "llm_metrics.csv"
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["method", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerow(metrics_row("llm", metrics))


def save_comparison_with_existing_metrics(output_dir, llm_metrics):
    """Append LLM metrics to existing baseline-vs-GA CSV when available."""
    output_dir = Path(output_dir)
    source_path = output_dir / "metrics_comparison.csv"
    target_path = output_dir / "method_comparison_with_llm.csv"
    rows = []

    if source_path.exists():
        with source_path.open("r", newline="", encoding="utf-8") as csv_file:
            rows.extend(csv.DictReader(csv_file))

    rows.append(metrics_row("llm", llm_metrics))

    with target_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["method", *METRIC_COLUMNS])
        writer.writeheader()
        writer.writerows(rows)


def metrics_row(method, metrics):
    """Return one CSV row for method metrics."""
    row = {"method": method}
    for key in METRIC_COLUMNS:
        row[key] = metrics.get(key, "")
    return row


def render_llm_outputs(graph, routes, output_dir):
    """Save LLM-specific maps and route images."""
    route_colors = config["route_colors"]
    output_dir = Path(output_dir)

    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        routes,
        route_colors,
        "LLM routes",
        output_dir / "12_llm_routes_colored.png",
        config,
    )
    plot_edge_load_map(
        graph,
        districts,
        stops,
        routes,
        "LLM edge load",
        output_dir / "13_llm_edge_load.png",
        config,
    )

    xlim, ylim = get_center_bounds(districts)
    plot_routes_with_offsets(
        graph,
        districts,
        stops,
        routes,
        route_colors,
        "LLM routes: center zoom",
        output_dir / "14_llm_center_zoom_colored.png",
        config,
        xlim=xlim,
        ylim=ylim,
    )
    plot_individual_routes(
        graph,
        districts,
        stops,
        routes,
        route_colors,
        output_dir,
        "llm",
        config,
    )


def print_summary(routes, metrics, output_dir):
    """Print concise LLM experiment results."""
    print("")
    print("LLM experiment completed.")
    print(f"Outputs: {Path(output_dir).resolve()}")
    print(f"LLM routes: {len(routes)}")
    print("")
    print("LLM metrics:")
    for key in METRIC_COLUMNS:
        value = metrics.get(key, 0.0)
        print(f"  {key}: {value:.4f}")


def save_text(path, text):
    """Write UTF-8 text."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    run_llm_experiment()
