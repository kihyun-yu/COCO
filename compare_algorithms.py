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
import os
from pathlib import Path
from dataclasses import dataclass

os.environ.setdefault("MPLCONFIGDIR", "/tmp/coco-matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/coco-cache")

import matplotlib
import numpy as np
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from algorithms import (
    COCOWithoutSlater2026,
    COCOWithoutSlaterConfig,
    YuNeelyWei2017,
    YuNeelyWei2017Doubling,
    YuNeelyWeiConfig,
)
from functions import QuadraticFunction
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
    comparator: np.ndarray
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
    path: Path,
    confidence_intervals: dict[str, tuple[list[float], list[float]]] | None = None,
) -> None:
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
    padded_y_min = y_min - padding
    padded_y_max = y_max + padding

    # Use only integer and half-integer tick values. This avoids labels such as
    # 17.37 while retaining a consistent scale across the four plots.
    y_step = _half_integer_tick_step(padded_y_max - padded_y_min)
    y_min = math.floor(padded_y_min / y_step) * y_step
    y_max = math.ceil(padded_y_max / y_step) * y_step
    if math.isclose(y_min, y_max):
        y_max = y_min + y_step

    # Align the horizontal axis and grid to exact multiples of 1,000.
    x_min = 0.0
    x_max = max(1000.0, math.ceil(float(rounds[-1]) / 1000.0) * 1000.0)

    y_ticks = np.arange(y_min, y_max + 0.5 * y_step, y_step)
    x_ticks = np.arange(0.0, x_max + 1.0, 1000.0)
    figure, axis = plt.subplots(figsize=(12, 7.5), dpi=150)
    for idx, (label, values) in enumerate(series.items()):
        color = colors[idx % len(colors)]
        if confidence_intervals is not None and label in confidence_intervals:
            lower, upper = confidence_intervals[label]
            axis.fill_between(rounds, lower, upper, color=color, alpha=0.14)
        axis.plot(
            rounds,
            values,
            color=color,
            linewidth=3.0,
            label=_plot_legend_label(label),
        )

    axis.set_xlim(x_min, x_max)
    axis.set_ylim(y_min, y_max)
    axis.set_xticks(x_ticks)
    axis.set_yticks(y_ticks)
    axis.set_yticklabels(
        [str(int(round(y))) if math.isclose(y, round(y)) else f"{y:.1f}" for y in y_ticks]
    )
    axis.set_xlabel("Number of interactions (round)", fontsize=24, labelpad=14)
    axis.set_ylabel(ylabel, fontsize=24, labelpad=14)
    axis.tick_params(axis="both", labelsize=20)
    axis.grid(True, color="#e5e7eb", linewidth=1.0)
    axis.axhline(0.0, color="#111827", linewidth=1.2)
    axis.legend(
        prop=FontProperties(family="monospace", size=20),
        loc="upper left",
        framealpha=0.88,
        edgecolor="#d1d5db",
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="jpeg", dpi=150, pil_kwargs={"quality": 95})
    plt.close(figure)


def _half_integer_tick_step(y_range: float) -> float:
    """Return a readable tick interval whose values stay on half-integers."""

    target = max(y_range / 6.0, 0.5)
    magnitude = 10.0 ** math.floor(math.log10(target))
    for multiplier in (1.0, 2.0, 5.0, 10.0):
        candidate = multiplier * magnitude
        if candidate >= target and candidate >= 0.5:
            return candidate
    return max(10.0 * magnitude, 0.5)


def _plot_legend_label(label: str) -> str:
    if "known T" in label:
        return "Yu2017 (known T)"
    if "doubling" in label:
        return "Yu2017 (doubling)"
    if "2026" in label:
        return "Ours (Algorithm 1)"
    return label


def save_result_plots(
    result: ComparisonResult,
    rounds_count: int,
    result_dir: Path,
    regret_dir: Path | None = None,
    violation_dir: Path | None = None,
    regret_filename: str = "regret.jpg",
    violation_filename: str = "constraint_violation.jpg",
) -> tuple[Path, Path]:
    regret_dir = result_dir if regret_dir is None else regret_dir
    violation_dir = result_dir if violation_dir is None else violation_dir
    regret_dir.mkdir(parents=True, exist_ok=True)
    violation_dir.mkdir(parents=True, exist_ok=True)
    rounds = np.arange(1, rounds_count + 1)
    regret_path = regret_dir / regret_filename
    violation_path = violation_dir / violation_filename
    save_plot(
        rounds,
        result.history.regret,
        "Regret",
        regret_path,
        None if result.confidence_intervals is None else result.confidence_intervals.regret,
    )
    save_plot(
        rounds,
        result.history.violation,
        "Constraint Violation",
        violation_path,
        None if result.confidence_intervals is None else result.confidence_intervals.violation,
    )
    return regret_path, violation_path


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
    complexity: str = "simple",
    show_progress: bool = False,
) -> ComparisonResult:
    if num_runs <= 0:
        raise ValueError("num_runs must be positive")
    progress = tqdm(
        total=num_runs * rounds_count,
        desc="COCO experiment",
        unit="round",
        disable=not show_progress,
        dynamic_ncols=True,
    )
    results = []
    try:
        for run_idx in range(num_runs):
            results.append(
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
                    progress=progress,
                )
            )
    finally:
        progress.close()
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
    progress: tqdm | None = None,
) -> ComparisonResult:
    if not 2 <= dim <= 10:
        raise ValueError("--dim must be between 2 and 10")

    loss_seed, constraint_seed = np.random.SeedSequence(seed).spawn(2)
    loss_rng = np.random.default_rng(loss_seed)
    constraint_rng = np.random.default_rng(constraint_seed)
    problem = make_problem(problem_name, dim=dim, slater_margin=slater_margin, complexity=complexity)
    box = problem.feasible_set
    x0 = np.full(dim, 0.5)

    constraint_counts = np.zeros(3 if problem_name == "slater" else 2, dtype=int)
    round_data = []
    for round_index in range(1, rounds_count + 1):
        loss, constraint, constraint_type = problem.sample_round(
            loss_rng, round_index, constraint_rng=constraint_rng
        )
        round_data.append((loss, constraint))
        constraint_counts[constraint_type] += 1
    comparator = _exact_empirical_comparator(
        [loss for loss, _ in round_data], problem_name, slater_margin
    )

    # Conservative bounds for this synthetic problem.
    L = max(math.sqrt(dim), math.sqrt(2.0))
    # A conservative absolute constraint-value bound on [0, 1]^dim,
    # including the bounded stochastic intercept shocks.
    G = float(dim + 2)
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
    for loss, constraint in round_data:
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
        if progress is not None:
            progress.update(1)

    round_wise_feasible = problem.round_wise_feasible_region_nonempty(constraint_counts)
    expected_constraint_at_comparator = problem.expected_constraint(comparator)
    return ComparisonResult(
        labels=labels,
        metrics=metrics,
        history=history,
        comparator=comparator.copy(),
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
        comparator=np.mean([result.comparator for result in results], axis=0),
        last_x=last_x,
        constraint_counts=np.sum([result.constraint_counts for result in results], axis=0),
        expected_constraint_at_comparator=float(
            np.mean([result.expected_constraint_at_comparator for result in results])
        ),
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


def _exact_empirical_comparator(
    losses: list[QuadraticFunction], problem_name: str, slater_margin: float
) -> np.ndarray:
    """Return the exact best fixed action for the realized simple losses."""

    if not losses or not all(isinstance(loss, QuadraticFunction) for loss in losses):
        raise ValueError(
            "exact empirical regret currently requires --complexity simple"
        )
    weights = np.asarray([loss.weight for loss in losses], dtype=float)
    if np.any(weights <= 0.0):
        raise ValueError("empirical comparator requires positive quadratic weights")
    centers = np.stack([loss.center for loss in losses])
    mean_center = np.average(centers, axis=0, weights=weights)

    if problem_name == "slater":
        comparator = np.clip(mean_center, 0.0, 1.0)
        comparator[:2] = _project_onto_nonnegative_simplex(
            comparator[:2], 0.275 / 0.35
        )
        return comparator
    if problem_name == "no-slater":
        return np.zeros_like(mean_center)
    if problem_name == "gradual-slater":
        return _project_onto_nonnegative_simplex(mean_center, slater_margin)
    raise ValueError(f"unknown problem: {problem_name}")


def _project_onto_nonnegative_simplex(vector: np.ndarray, radius: float) -> np.ndarray:
    """Project onto {x >= 0: sum(x) <= radius}."""

    if radius < 0.0:
        raise ValueError("simplex radius must be nonnegative")
    positive = np.maximum(np.asarray(vector, dtype=float), 0.0)
    if float(np.sum(positive)) <= radius:
        return positive
    if radius == 0.0:
        return np.zeros_like(positive)

    sorted_values = np.sort(positive)[::-1]
    cumulative = np.cumsum(sorted_values) - radius
    indices = np.arange(1, positive.size + 1)
    valid = sorted_values - cumulative / indices > 0.0
    rho = int(np.flatnonzero(valid)[-1])
    threshold = cumulative[rho] / float(rho + 1)
    return np.maximum(positive - threshold, 0.0)
