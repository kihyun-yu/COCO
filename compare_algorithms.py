"""Compare the two implemented COCO algorithms on a synthetic problem.

The stochastic constraints are intentionally chosen to favor algorithms that use
feasibility in expectation.  Realized constraints are sampled from conflicting
convex families with time-varying probabilities:

    type 0: x[0] <= 0.4
    type 1: x[1] <= 0.4
    type 2: x[0] + x[1] >= 0.85

If all three types appear, no point in X=[0,1]^d is round-wise feasible.
However, because type 2 is rare in the long-run mixture, the expected
constraint is feasible:

    E[g_t(x)] = 0.35 * (x[0] + x[1]) - 0.275 <= 0.

Use ``main.py`` to run the experiment from the command line.
"""

from __future__ import annotations

import math
from pathlib import Path
from dataclasses import dataclass
from html import escape

import numpy as np

from algorithms import (
    COCOWithoutSlater2026,
    COCOWithoutSlaterConfig,
    YuNeelyWei2017,
    YuNeelyWei2017Doubling,
    YuNeelyWeiConfig,
)
from problems import make_problem


@dataclass
class Metrics:
    loss: float = 0.0
    comparator_loss: float = 0.0
    violation: float = 0.0

    @property
    def regret(self) -> float:
        return self.loss - self.comparator_loss


@dataclass
class History:
    regret: dict[str, list[float]]
    violation: dict[str, list[float]]


@dataclass
class ConfidenceIntervals:
    regret: dict[str, tuple[list[float], list[float]]]
    violation: dict[str, tuple[list[float], list[float]]]


@dataclass
class ComparisonResult:
    labels: list[str]
    metrics: dict[str, Metrics]
    history: History
    last_x: dict[str, np.ndarray]
    constraint_counts: np.ndarray
    expected_constraint_at_comparator: float
    round_wise_feasible_region_nonempty: bool
    problem_name: str
    num_runs: int = 1
    confidence_intervals: ConfidenceIntervals | None = None


def summarize(name: str, metrics: Metrics, last_x: np.ndarray) -> str:
    return (
        f"{name:24s} "
        f"regret={metrics.regret:10.4f} "
        f"violation={metrics.violation:10.4f} "
        f"loss={metrics.loss:10.4f} "
        f"last_x={np.array2string(last_x, precision=3)}"
    )


def save_plot(
    rounds: np.ndarray,
    series: dict[str, list[float]],
    ylabel: str,
    title: str,
    path: Path,
    confidence_intervals: dict[str, tuple[list[float], list[float]]] | None = None,
) -> None:
    width, height = 960, 600
    left, right, top, bottom = 90, 30, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

    all_values = [float(v) for values in series.values() for v in values]
    if confidence_intervals is not None:
        for lower, upper in confidence_intervals.values():
            all_values.extend(float(v) for v in lower)
            all_values.extend(float(v) for v in upper)
    y_min = min(all_values + [0.0])
    y_max = max(all_values + [0.0])
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0
    padding = 0.05 * (y_max - y_min)
    y_min -= padding
    y_max += padding
    x_min = float(rounds[0])
    x_max = float(rounds[-1])

    def x_coord(x: float) -> float:
        if math.isclose(x_min, x_max):
            return left + plot_width / 2.0
        return left + (x - x_min) / (x_max - x_min) * plot_width

    def y_coord(y: float) -> float:
        return top + (y_max - y) / (y_max - y_min) * plot_height

    y_ticks = np.linspace(y_min, y_max, 6)
    x_ticks = np.linspace(x_min, x_max, 6)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="bold">{escape(title)}</text>',
    ]

    for y in y_ticks:
        yy = y_coord(float(y))
        elements.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width - right}" y2="{yy:.2f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left - 10}" y="{yy + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{y:.2f}</text>')

    for x in x_ticks:
        xx = x_coord(float(x))
        elements.append(f'<line x1="{xx:.2f}" y1="{top}" x2="{xx:.2f}" y2="{height - bottom}" stroke="#f3f4f6"/>')
        elements.append(f'<text x="{xx:.2f}" y="{height - bottom + 22}" text-anchor="middle" font-family="Arial" font-size="12">{int(round(x))}</text>')

    zero_y = y_coord(0.0)
    if top <= zero_y <= height - bottom:
        elements.append(f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width - right}" y2="{zero_y:.2f}" stroke="#111827" stroke-width="1"/>')

    elements.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" stroke-width="1.2"/>')
    elements.append(f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" stroke-width="1.2"/>')
    elements.append(f'<text x="{width / 2}" y="{height - 25}" text-anchor="middle" font-family="Arial" font-size="14">Number of interactions (round)</text>')
    elements.append(f'<text x="24" y="{height / 2}" transform="rotate(-90 24 {height / 2})" text-anchor="middle" font-family="Arial" font-size="14">{escape(ylabel)}</text>')

    for idx, (label, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        if confidence_intervals is not None and label in confidence_intervals:
            lower, upper = confidence_intervals[label]
            upper_points = [
                f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}"
                for x, y in zip(rounds, upper)
            ]
            lower_points = [
                f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}"
                for x, y in zip(rounds[::-1], list(lower)[::-1])
            ]
            band_points = " ".join(upper_points + lower_points)
            elements.append(f'<polygon points="{band_points}" fill="{color}" opacity="0.14"/>')
        points = " ".join(
            f"{x_coord(float(x)):.2f},{y_coord(float(y)):.2f}"
            for x, y in zip(rounds, values)
        )
        elements.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.4" points="{points}"/>')
        legend_y = top + 22 + idx * 22
        legend_x = left + 20
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="{color}" stroke-width="2.4"/>')
        elements.append(f'<text x="{legend_x + 36}" y="{legend_y + 4}" font-family="Arial" font-size="13">{escape(label)}</text>')

    elements.append("</svg>")
    path.write_text("\n".join(elements), encoding="utf-8")


def save_result_plots(
    result: ComparisonResult,
    rounds_count: int,
    result_dir: Path,
    regret_dir: Path | None = None,
    violation_dir: Path | None = None,
    regret_filename: str = "regret.svg",
    violation_filename: str = "constraint_violation.svg",
    normalized_regret_filename: str = "normalized_regret.svg",
    normalized_violation_filename: str = "normalized_constraint_violation.svg",
) -> tuple[Path, Path, Path, Path]:
    regret_dir = result_dir if regret_dir is None else regret_dir
    violation_dir = result_dir if violation_dir is None else violation_dir
    regret_dir.mkdir(parents=True, exist_ok=True)
    violation_dir.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(1, rounds_count + 1)
    regret_path = regret_dir / regret_filename
    violation_path = violation_dir / violation_filename
    normalized_regret_path = regret_dir / normalized_regret_filename
    normalized_violation_path = violation_dir / normalized_violation_filename
    save_plot(
        rounds,
        result.history.regret,
        "Cumulative regret",
        "Regret vs. interactions",
        regret_path,
        None if result.confidence_intervals is None else result.confidence_intervals.regret,
    )
    save_plot(
        rounds,
        result.history.violation,
        "Cumulative constraint violation",
        "Constraint violation vs. interactions",
        violation_path,
        None if result.confidence_intervals is None else result.confidence_intervals.violation,
    )
    save_plot(
        rounds,
        _normalize_series(result.history.regret, rounds),
        "Average regret",
        "Average regret vs. interactions",
        normalized_regret_path,
        None
        if result.confidence_intervals is None
        else _normalize_confidence_intervals(result.confidence_intervals.regret, rounds),
    )
    save_plot(
        rounds,
        _normalize_series(result.history.violation, rounds),
        "Average constraint violation",
        "Average constraint violation vs. interactions",
        normalized_violation_path,
        None
        if result.confidence_intervals is None
        else _normalize_confidence_intervals(result.confidence_intervals.violation, rounds),
    )
    return regret_path, violation_path, normalized_regret_path, normalized_violation_path


def _normalize_series(series: dict[str, list[float]], rounds: np.ndarray) -> dict[str, list[float]]:
    return {
        label: (np.asarray(values, dtype=float) / rounds).tolist()
        for label, values in series.items()
    }


def _normalize_confidence_intervals(
    confidence_intervals: dict[str, tuple[list[float], list[float]]],
    rounds: np.ndarray,
) -> dict[str, tuple[list[float], list[float]]]:
    normalized: dict[str, tuple[list[float], list[float]]] = {}
    for label, (lower, upper) in confidence_intervals.items():
        normalized[label] = (
            (np.asarray(lower, dtype=float) / rounds).tolist(),
            (np.asarray(upper, dtype=float) / rounds).tolist(),
        )
    return normalized


def run_comparison(
    rounds_count: int,
    dim: int,
    seed: int,
    num_runs: int = 10,
    high_probability: bool = False,
    practical_gamma_scale: float = 20.0,
    practical_regularizer_scale: float = 0.1,
    problem_name: str = "slater",
    slater_margin: float = 0.0,
    complexity: str = "complicated",
) -> ComparisonResult:
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")
    results = [
        _run_single_comparison(
            rounds_count=rounds_count,
            dim=dim,
            seed=seed + run_idx,
            high_probability=high_probability,
            practical_gamma_scale=practical_gamma_scale,
            practical_regularizer_scale=practical_regularizer_scale,
            problem_name=problem_name,
            slater_margin=slater_margin,
            complexity=complexity,
        )
        for run_idx in range(num_runs)
    ]
    if num_runs == 1:
        return results[0]
    return _aggregate_results(results)


def _run_single_comparison(
    rounds_count: int,
    dim: int,
    seed: int,
    high_probability: bool,
    practical_gamma_scale: float,
    practical_regularizer_scale: float,
    problem_name: str,
    slater_margin: float,
    complexity: str,
) -> ComparisonResult:
    if dim < 2:
        raise ValueError("--dim must be at least 2 for the conflicting stochastic constraints")

    rng = np.random.default_rng(seed)
    problem = make_problem(problem_name, dim=dim, slater_margin=slater_margin, complexity=complexity)
    box = problem.feasible_set
    x0 = np.full(dim, 0.5)
    comparator = problem.comparator

    # Conservative bounds for this synthetic problem.
    L = max(math.sqrt(dim), math.sqrt(2.0))
    G = 1.0
    D = box.diameter

    alg_2017_known_t = YuNeelyWei2017(
        feasible_set=box,
        x=x0,
        num_constraints=1,
        config=YuNeelyWeiConfig(V=math.sqrt(rounds_count), alpha=rounds_count),
    )
    alg_2017_doubling = YuNeelyWei2017Doubling(
        feasible_set=box,
        x=x0,
        num_constraints=1,
    )
    alg_2026_practical = COCOWithoutSlater2026(
        feasible_set=box,
        x=x0,
        config=COCOWithoutSlaterConfig(
            L=L,
            G=G,
            D=D,
            high_probability=high_probability,
            gamma_scale=practical_gamma_scale,
            regularizer_scale=practical_regularizer_scale,
        ),
    )
    labels = [
        "Yu-Neely-Wei 2017 (known T)",
        "Yu-Neely-Wei 2017 (doubling)",
        (
            "Yu-Lee-Lee 2026 "
            f"(practical, gamma x{practical_gamma_scale:g}, reg x{practical_regularizer_scale:g})"
        ),
    ]
    metrics = {label: Metrics() for label in labels}
    history = History(
        regret={label: [] for label in labels},
        violation={label: [] for label in labels},
    )
    constraint_counts = np.zeros(3 if problem_name == "slater" else 2, dtype=int)

    for round_index in range(1, rounds_count + 1):
        loss, constraint, constraint_type = problem.sample_round(rng, round_index)
        constraint_counts[constraint_type] += 1
        outputs = {
            labels[0]: alg_2017_known_t.step(loss, [constraint]),
            labels[1]: alg_2017_doubling.step(loss, [constraint]),
            labels[2]: alg_2026_practical.step(loss, constraint),
        }

        comparator_loss = loss.value(comparator)
        for label, out in outputs.items():
            metrics[label].loss += float(out["loss"])
            metrics[label].comparator_loss += comparator_loss
            if "constraints" in out:
                metrics[label].violation += float(out["constraints"][0])
            else:
                metrics[label].violation += float(out["constraint"])
            history.regret[label].append(metrics[label].regret)
            history.violation[label].append(metrics[label].violation)

    round_wise_feasible = problem.round_wise_feasible_region_nonempty(constraint_counts)
    expected_constraint_at_comparator = problem.expected_constraint(comparator)
    return ComparisonResult(
        labels=labels,
        metrics=metrics,
        history=history,
        last_x={
            labels[0]: alg_2017_known_t.x.copy(),
            labels[1]: alg_2017_doubling.x.copy(),
            labels[2]: alg_2026_practical.x.copy(),
        },
        constraint_counts=constraint_counts,
        expected_constraint_at_comparator=expected_constraint_at_comparator,
        round_wise_feasible_region_nonempty=round_wise_feasible,
        problem_name=problem_name,
    )


def _aggregate_results(results: list[ComparisonResult]) -> ComparisonResult:
    first = results[0]
    labels = first.labels
    num_runs = len(results)
    regret_history: dict[str, list[float]] = {}
    violation_history: dict[str, list[float]] = {}
    regret_ci: dict[str, tuple[list[float], list[float]]] = {}
    violation_ci: dict[str, tuple[list[float], list[float]]] = {}
    metrics: dict[str, Metrics] = {}
    last_x: dict[str, np.ndarray] = {}

    for label in labels:
        regret_samples = np.array([result.history.regret[label] for result in results], dtype=float)
        violation_samples = np.array([result.history.violation[label] for result in results], dtype=float)
        regret_mean, regret_lower, regret_upper = _mean_and_ci(regret_samples)
        violation_mean, violation_lower, violation_upper = _mean_and_ci(violation_samples)
        regret_history[label] = regret_mean.tolist()
        violation_history[label] = violation_mean.tolist()
        regret_ci[label] = (regret_lower.tolist(), regret_upper.tolist())
        violation_ci[label] = (violation_lower.tolist(), violation_upper.tolist())
        metrics[label] = Metrics(
            loss=float(np.mean([result.metrics[label].loss for result in results])),
            comparator_loss=float(np.mean([result.metrics[label].comparator_loss for result in results])),
            violation=float(np.mean([result.metrics[label].violation for result in results])),
        )
        last_x[label] = np.mean([result.last_x[label] for result in results], axis=0)

    return ComparisonResult(
        labels=labels,
        metrics=metrics,
        history=History(regret=regret_history, violation=violation_history),
        last_x=last_x,
        constraint_counts=np.sum([result.constraint_counts for result in results], axis=0),
        expected_constraint_at_comparator=first.expected_constraint_at_comparator,
        round_wise_feasible_region_nonempty=all(
            result.round_wise_feasible_region_nonempty for result in results
        ),
        problem_name=first.problem_name,
        num_runs=num_runs,
        confidence_intervals=ConfidenceIntervals(regret=regret_ci, violation=violation_ci),
    )


def _mean_and_ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(samples, axis=0)
    stderr = np.std(samples, axis=0, ddof=1) / math.sqrt(samples.shape[0])
    half_width = 1.96 * stderr
    return mean, mean - half_width, mean + half_width
