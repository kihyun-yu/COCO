"""Entry point for running the COCO algorithm comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import threading

import numpy as np

_MKL_SSE_WARNING = "Intel MKL WARNING: Support of Intel(R) Streaming SIMD Extensions 4.2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--dim", type=int, choices=range(2, 11), default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem", choices=["slater", "no-slater", "gradual-slater"], default="slater")
    parser.add_argument("--slater-margin", type=float, default=0.0)
    parser.add_argument("--loss-switch-interval", type=int, default=30)
    parser.add_argument(
        "--loss-schedule",
        choices=["complementary", "sinusoidal"],
        default="complementary",
    )
    parser.add_argument("--loss-rotation-period", type=int, default=200)
    parser.add_argument("--high-probability", action="store_true")
    parser.add_argument("--practical-gamma-scale", type=float, default=1.0)
    parser.add_argument("--practical-regularizer-scale", type=float, default=1.0)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--regret-dir", type=Path, default=None)
    parser.add_argument("--violation-dir", type=Path, default=None)
    parser.add_argument("--trajectory-dir", type=Path, default=None)
    parser.add_argument("--regret-filename", default="regret.jpg")
    parser.add_argument("--violation-filename", default="constraint_violation.jpg")
    parser.add_argument("--trajectory-filename", default="decision_trajectory.jpg")
    return parser.parse_args()


def import_comparison_tools():
    """Import NumPy-backed experiment code while filtering one noisy MKL warning."""

    if os.environ.get("COCO_SHOW_MKL_WARNINGS") == "1":
        from compare_algorithms import run_comparison, save_result_plots, summarize

        return run_comparison, save_result_plots, summarize

    read_fd, write_fd = os.pipe()
    old_stderr_fd = os.dup(2)
    captured: list[bytes] = []

    def read_stderr() -> None:
        with os.fdopen(read_fd, "rb") as stream:
            for chunk in iter(lambda: stream.read(8192), b""):
                captured.append(chunk)

    reader = threading.Thread(target=read_stderr)
    reader.start()
    try:
        sys.stderr.flush()
        os.dup2(write_fd, 2)
        from compare_algorithms import run_comparison, save_result_plots, summarize
    finally:
        sys.stderr.flush()
        os.dup2(old_stderr_fd, 2)
        os.close(write_fd)
        os.close(old_stderr_fd)
        reader.join()

    stderr_text = b"".join(captured).decode(errors="replace")
    for line in stderr_text.splitlines(keepends=True):
        if _MKL_SSE_WARNING not in line:
            sys.stderr.write(line)
    return run_comparison, save_result_plots, summarize


def main() -> None:
    args = parse_args()
    run_comparison, save_result_plots, summarize = import_comparison_tools()
    result = run_comparison(
        rounds_count=args.rounds,
        dim=args.dim,
        seed=args.seed,
        num_runs=args.runs,
        high_probability=args.high_probability,
        practical_gamma_scale=args.practical_gamma_scale,
        practical_regularizer_scale=args.practical_regularizer_scale,
        problem_name=args.problem,
        slater_margin=args.slater_margin,
        loss_switch_interval=args.loss_switch_interval,
        loss_schedule=args.loss_schedule,
        loss_rotation_period=args.loss_rotation_period,
        show_progress=not args.no_progress,
    )
    regret_path, violation_path, trajectory_paths = save_result_plots(
        result,
        args.rounds,
        args.result_dir,
        regret_dir=args.regret_dir,
        violation_dir=args.violation_dir,
        trajectory_dir=args.trajectory_dir,
        regret_filename=args.regret_filename,
        violation_filename=args.violation_filename,
        trajectory_filename=args.trajectory_filename,
    )

    counts = result.constraint_counts
    print(
        f"rounds={args.rounds} runs={args.runs} dim={args.dim} seed={args.seed} "
        f"problem={args.problem} slater_margin={args.slater_margin} "
        f"loss_schedule={args.loss_schedule} "
        f"loss_switch_interval={args.loss_switch_interval} "
        f"loss_rotation_period={args.loss_rotation_period}"
    )
    if args.problem == "slater":
        print(
            "constraint_counts="
            f"x0<=0.4:{counts[0]} "
            f"x1<=0.4:{counts[1]} "
            f"x0+x1>=0.85:{counts[2]}"
        )
    elif args.problem == "no-slater":
        print(
            "constraint_counts="
            f"noise_negative:{counts[0]} "
            f"noise_positive_roundwise_empty:{counts[1]}"
        )
    else:
        print(
            "constraint_counts="
            f"noise_negative:{counts[0]} "
            f"noise_positive:{counts[1]}"
        )
    print(f"expected_constraint_at_comparator={result.expected_constraint_at_comparator:.6f}")
    print(
        "empirical_comparator="
        f"{np.array2string(result.comparator, precision=6)}"
    )
    print(f"round_wise_feasible_region_nonempty={result.round_wise_feasible_region_nonempty}")
    for label in result.labels:
        print(summarize(label, result.metrics[label], result.last_x[label]))
    print(f"saved_regret_plot={regret_path}")
    print(f"saved_constraint_violation_plot={violation_path}")
    for trajectory_path in trajectory_paths:
        print(f"saved_decision_trajectory_plot={trajectory_path}")


if __name__ == "__main__":
    main()
