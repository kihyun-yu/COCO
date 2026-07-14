"""Synthetic COCO problem settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions import (
    AffineFunction,
    ConvexFunction,
    QuadraticFunction,
)
from sets import BoxSet


CONSTRAINT_INTERCEPT_NOISE_SCALE = 0.16
MIN_SOLUTION_DIM = 2
MAX_SOLUTION_DIM = 10
DEFAULT_SOLUTION_DIM = 5
EXTRA_COORDINATE_CENTER = 0.5
SLATER_NOISE_MAGNITUDE = 0.35
DEFAULT_LOSS_SWITCH_INTERVAL = 30
DEFAULT_LOSS_ROTATION_PERIOD = 200
LOSS_SCHEDULES = {"complementary", "sinusoidal"}


@dataclass(frozen=True)
class ConflictingStochasticConstraints:
    """A scalar stochastic-constraint problem with no round-wise comparator.

    Each round samples one of three constraint families using time-varying
    probabilities centered around the base mixture:
    - x[0] <= 0.4
    - x[1] <= 0.4
    - x[0] + x[1] >= 0.85

    The intersection of all three realized feasible regions is empty, but the
    long-run expected feasible region is non-empty:
        E[g_t(x)] = 0.35 * (x[0] + x[1]) - 0.275 <= 0.
    """

    dim: int = DEFAULT_SOLUTION_DIM
    probabilities: tuple[float, float, float] = (0.45, 0.45, 0.10)
    loss_switch_interval: int = DEFAULT_LOSS_SWITCH_INTERVAL
    loss_schedule: str = "complementary"
    loss_rotation_period: int = DEFAULT_LOSS_ROTATION_PERIOD

    def __post_init__(self) -> None:
        _validate_dim(self.dim)
        _validate_loss_switch_interval(self.loss_switch_interval)
        _validate_loss_schedule(self.loss_schedule, self.loss_rotation_period)
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("probabilities must sum to 1")

    @property
    def feasible_set(self) -> BoxSet:
        return BoxSet(lower=np.zeros(self.dim), upper=np.ones(self.dim))

    @property
    def comparator(self) -> np.ndarray:
        x = np.full(self.dim, EXTRA_COORDINATE_CENTER)
        boundary_value = 0.275 / 0.7
        x[0] = boundary_value
        x[1] = boundary_value
        return x

    def expected_constraint(self, x: np.ndarray) -> float:
        return float(0.35 * (x[0] + x[1]) - 0.275)

    def round_wise_feasible_region_nonempty(self, constraint_counts: np.ndarray) -> bool:
        return not np.all(np.asarray(constraint_counts) > 0)

    def constraint_probabilities(self, round_index: int) -> np.ndarray:
        """Time-varying mixture over constraint families.

        The sampler alternates between smooth drift and short burst periods. It
        is more realistic than a fixed categorical distribution, while staying
        close to the base probabilities over long horizons.
        """

        t = max(1, int(round_index))
        base = np.asarray(self.probabilities, dtype=float)
        slow_shift = 0.16 * np.sin(2.0 * np.pi * t / 211.0)
        fast_x = 0.07 * np.sin(2.0 * np.pi * t / 37.0)
        fast_y = 0.07 * np.cos(2.0 * np.pi * t / 43.0)
        drift = np.array([
            slow_shift + fast_x,
            -slow_shift + fast_y,
            -fast_x - fast_y,
        ])

        burst_phase = (t // 55) % 5
        burst = np.zeros(3)
        if burst_phase == 1:
            burst = np.array([0.22, -0.12, -0.10])
        elif burst_phase == 2:
            burst = np.array([-0.12, 0.22, -0.10])
        elif burst_phase == 3:
            burst = np.array([-0.10, -0.10, 0.20])

        probabilities = np.clip(base + drift + burst, 0.03, 0.90)
        return probabilities / float(np.sum(probabilities))

    def sample_round(
        self,
        rng: np.random.Generator,
        round_index: int = 0,
        constraint_rng: np.random.Generator | None = None,
    ) -> tuple[ConvexFunction, ConvexFunction, int]:
        loss = _scheduled_loss(
            rng,
            self.dim,
            round_index,
            self.loss_switch_interval,
            self.loss_schedule,
            self.loss_rotation_period,
        )
        constraint_rng = rng if constraint_rng is None else constraint_rng
        intercept_noise = _symmetric_uniform_noise(
            constraint_rng, CONSTRAINT_INTERCEPT_NOISE_SCALE
        )

        constraint_type = int(
            constraint_rng.choice(3, p=self.constraint_probabilities(round_index))
        )
        if constraint_type == 0:
            a = np.zeros(self.dim)
            a[0] = 1.0
            constraint = AffineFunction(a=a, b=-0.4 + intercept_noise)
        elif constraint_type == 1:
            a = np.zeros(self.dim)
            a[1] = 1.0
            constraint = AffineFunction(a=a, b=-0.4 + intercept_noise)
        else:
            a = np.zeros(self.dim)
            a[0] = -1.0
            a[1] = -1.0
            constraint = AffineFunction(a=a, b=0.85 + intercept_noise)
        return loss, constraint, constraint_type


@dataclass(frozen=True)
class NoSlaterStochasticConstraints:
    """Stochastic COCO problem where Slater's condition fails.

    The expected constraint is
        E[g_t(x)] = sum_i x[i] <= 0
    on X = [0, 1]^d.  Thus the expected feasible set is non-empty, but it is
    the singleton {(0, 0, ...)}, so no point satisfies E[g_t(x)] < 0.

    Realized constraints are g_t(x) = sum_i x[i] + noise_t with zero-mean
    bounded noise.  Whenever noise_t > 0, the round-wise feasible set is empty.
    """

    dim: int = DEFAULT_SOLUTION_DIM
    noise_magnitude: float = SLATER_NOISE_MAGNITUDE
    loss_switch_interval: int = DEFAULT_LOSS_SWITCH_INTERVAL
    loss_schedule: str = "complementary"
    loss_rotation_period: int = DEFAULT_LOSS_ROTATION_PERIOD

    def __post_init__(self) -> None:
        _validate_dim(self.dim)
        _validate_loss_switch_interval(self.loss_switch_interval)
        _validate_loss_schedule(self.loss_schedule, self.loss_rotation_period)
        if self.noise_magnitude <= 0:
            raise ValueError("noise_magnitude must be positive")

    @property
    def feasible_set(self) -> BoxSet:
        return BoxSet(lower=np.zeros(self.dim), upper=np.ones(self.dim))

    @property
    def comparator(self) -> np.ndarray:
        return np.zeros(self.dim)

    def expected_constraint(self, x: np.ndarray) -> float:
        return float(np.sum(x))

    def round_wise_feasible_region_nonempty(self, constraint_counts: np.ndarray) -> bool:
        return int(np.asarray(constraint_counts)[1]) == 0

    def sample_round(
        self,
        rng: np.random.Generator,
        round_index: int = 0,
        constraint_rng: np.random.Generator | None = None,
    ) -> tuple[ConvexFunction, ConvexFunction, int]:
        loss = _scheduled_loss(
            rng,
            self.dim,
            round_index,
            self.loss_switch_interval,
            self.loss_schedule,
            self.loss_rotation_period,
        )
        constraint_rng = rng if constraint_rng is None else constraint_rng

        noise = _symmetric_uniform_noise(constraint_rng, self.noise_magnitude)
        constraint = _margin_constraint(self.dim, margin=0.0, noise=noise)
        return loss, constraint, int(noise > 0.0)


@dataclass(frozen=True)
class GradualSlaterStochasticConstraints:
    """Stochastic COCO problem with controllable Slater margin.

    The expected constraint is
        E[g_t(x)] = sum_i x[i] - margin <= 0.

    If margin > 0, Slater's condition holds because x = 0 has expected
    constraint -margin.  If margin = 0, this reduces to the no-Slater singleton
    feasible set.  Decreasing margin therefore makes Slater-ness gradual.
    """

    dim: int = DEFAULT_SOLUTION_DIM
    margin: float = 0.1
    noise_magnitude: float = SLATER_NOISE_MAGNITUDE
    loss_switch_interval: int = DEFAULT_LOSS_SWITCH_INTERVAL
    loss_schedule: str = "complementary"
    loss_rotation_period: int = DEFAULT_LOSS_ROTATION_PERIOD

    def __post_init__(self) -> None:
        _validate_dim(self.dim)
        _validate_loss_switch_interval(self.loss_switch_interval)
        _validate_loss_schedule(self.loss_schedule, self.loss_rotation_period)
        if self.margin < 0:
            raise ValueError("margin must be nonnegative")
        if self.noise_magnitude <= 0:
            raise ValueError("noise_magnitude must be positive")

    @property
    def feasible_set(self) -> BoxSet:
        return BoxSet(lower=np.zeros(self.dim), upper=np.ones(self.dim))

    @property
    def comparator(self) -> np.ndarray:
        return np.full(self.dim, self.margin / self.dim)

    def expected_constraint(self, x: np.ndarray) -> float:
        return float(np.sum(x) - self.margin)

    def round_wise_feasible_region_nonempty(self, constraint_counts: np.ndarray) -> bool:
        if self.margin >= self.noise_magnitude:
            return True
        return int(np.asarray(constraint_counts)[1]) == 0

    def sample_round(
        self,
        rng: np.random.Generator,
        round_index: int = 0,
        constraint_rng: np.random.Generator | None = None,
    ) -> tuple[ConvexFunction, ConvexFunction, int]:
        loss = _scheduled_loss(
            rng,
            self.dim,
            round_index,
            self.loss_switch_interval,
            self.loss_schedule,
            self.loss_rotation_period,
        )
        constraint_rng = rng if constraint_rng is None else constraint_rng

        noise = _symmetric_uniform_noise(constraint_rng, self.noise_magnitude)
        constraint = _margin_constraint(self.dim, margin=self.margin, noise=noise)
        return loss, constraint, int(noise > self.margin)


def make_problem(
    name: str,
    dim: int,
    slater_margin: float = 0.0,
    loss_switch_interval: int = DEFAULT_LOSS_SWITCH_INTERVAL,
    loss_schedule: str = "complementary",
    loss_rotation_period: int = DEFAULT_LOSS_ROTATION_PERIOD,
) -> ConflictingStochasticConstraints | NoSlaterStochasticConstraints | GradualSlaterStochasticConstraints:
    if name == "slater":
        return ConflictingStochasticConstraints(
            dim=dim,
            loss_switch_interval=loss_switch_interval,
            loss_schedule=loss_schedule,
            loss_rotation_period=loss_rotation_period,
        )
    if name == "no-slater":
        return NoSlaterStochasticConstraints(
            dim=dim,
            loss_switch_interval=loss_switch_interval,
            loss_schedule=loss_schedule,
            loss_rotation_period=loss_rotation_period,
        )
    if name == "gradual-slater":
        return GradualSlaterStochasticConstraints(
            dim=dim,
            margin=slater_margin,
            loss_switch_interval=loss_switch_interval,
            loss_schedule=loss_schedule,
            loss_rotation_period=loss_rotation_period,
        )
    raise ValueError(f"unknown problem: {name}")


def _validate_dim(dim: int) -> None:
    if not MIN_SOLUTION_DIM <= dim <= MAX_SOLUTION_DIM:
        raise ValueError(
            f"dim must be between {MIN_SOLUTION_DIM} and {MAX_SOLUTION_DIM}"
        )


def _validate_loss_switch_interval(loss_switch_interval: int) -> None:
    if loss_switch_interval <= 0:
        raise ValueError("loss_switch_interval must be positive")


def _validate_loss_schedule(loss_schedule: str, loss_rotation_period: int) -> None:
    if loss_schedule not in LOSS_SCHEDULES:
        raise ValueError(
            f"loss_schedule must be one of {sorted(LOSS_SCHEDULES)}"
        )
    if loss_rotation_period <= 0:
        raise ValueError("loss_rotation_period must be positive")


def _scheduled_loss(
    rng: np.random.Generator,
    dim: int,
    round_index: int,
    loss_switch_interval: int,
    loss_schedule: str,
    loss_rotation_period: int,
) -> ConvexFunction:
    center = _scheduled_center(
        rng,
        dim,
        round_index,
        loss_switch_interval,
        loss_schedule,
        loss_rotation_period,
    )
    return QuadraticFunction(center=center)


def _scheduled_center(
    rng: np.random.Generator,
    dim: int,
    round_index: int,
    loss_switch_interval: int,
    loss_schedule: str,
    loss_rotation_period: int,
) -> np.ndarray:
    del rng
    t = max(1, int(round_index))
    if loss_schedule == "sinusoidal":
        angle = 2.0 * np.pi * (t - 1) / loss_rotation_period
        if dim == 2:
            return np.array([
                0.5 + 0.3 * np.cos(angle),
                0.5 + 0.3 * np.sin(angle),
            ])
        phases = 2.0 * np.pi * np.arange(dim) / dim
        return 0.5 + 0.3 * np.cos(angle + phases)

    # Complementary schedule retained for piecewise-constant experiments.
    block = (t - 1) // loss_switch_interval
    if dim == 2:
        return (
            np.array([0.8, 0.2])
            if block % 2 == 0
            else np.array([0.2, 0.8])
        )

    phase = block % (2 * dim)
    coordinate = phase // 2
    center = np.full(dim, 0.2)
    center[coordinate] = 0.8
    return center if phase % 2 == 0 else 1.0 - center


def _margin_constraint(dim: int, margin: float, noise: float) -> ConvexFunction:
    a = np.ones(dim)
    return AffineFunction(a=a, b=noise - margin)


def _symmetric_uniform_noise(rng: np.random.Generator, magnitude: float) -> float:
    """Sample simple bounded zero-mean noise from Uniform[-magnitude, magnitude]."""

    return float(rng.uniform(-magnitude, magnitude))
