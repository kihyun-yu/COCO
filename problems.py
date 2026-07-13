"""Synthetic COCO problem settings."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from functions import (
    AffineFunction,
    ConvexFunction,
    LogSumExpAffineFunction,
    PSDQuadraticFunction,
    QuadraticFunction,
    SmoothAbsFunction,
    SumFunction,
)
from sets import BoxSet


CONSTRAINT_INTERCEPT_NOISE_SCALE = 0.16
MIN_SOLUTION_DIM = 2
MAX_SOLUTION_DIM = 10
DEFAULT_SOLUTION_DIM = 5
EXTRA_COORDINATE_CENTER = 0.5
SLATER_NOISE_MAGNITUDE = 0.35
LOSS_LINEAR_NOISE_SCALE = 0.42
LOSS_LINEAR_SHOCK_PROBABILITY = 0.26
LOSS_LINEAR_SHOCK_MULTIPLIER = 3.5


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
        E[g_t(x)] = 0.35 * (x[0] + x[1]) + nonlinear terms - 0.275 <= 0.
    """

    dim: int = DEFAULT_SOLUTION_DIM
    probabilities: tuple[float, float, float] = (0.45, 0.45, 0.10)
    complexity: str = "simple"

    def __post_init__(self) -> None:
        _validate_dim(self.dim)
        if not np.isclose(sum(self.probabilities), 1.0):
            raise ValueError("probabilities must sum to 1")
        _validate_complexity(self.complexity)

    @property
    def feasible_set(self) -> BoxSet:
        return BoxSet(lower=np.zeros(self.dim), upper=np.ones(self.dim))

    @property
    def comparator(self) -> np.ndarray:
        x = np.full(self.dim, EXTRA_COORDINATE_CENTER)
        boundary_value = (
            0.275 / 0.7
            if self.complexity == "simple"
            else _conflicting_symmetric_boundary()
        )
        x[0] = boundary_value
        x[1] = boundary_value
        return x

    def expected_constraint(self, x: np.ndarray) -> float:
        if self.complexity == "simple":
            return float(0.35 * (x[0] + x[1]) - 0.275)
        return float(
            0.35 * (x[0] + x[1])
            + 0.045 * (x[0] * x[0] + x[1] * x[1])
            + 0.01 * (x[0] - x[1]) ** 2
            - 0.275
        )

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
        loss = _scheduled_loss(rng, self.dim, round_index, self.complexity)
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
            if self.complexity == "simple":
                return loss, AffineFunction(a=a, b=-0.4 + intercept_noise), constraint_type
            H = np.zeros((self.dim, self.dim))
            H[1, 1] = 0.2
            constraint = PSDQuadraticFunction(
                H=H, center=np.zeros(self.dim), a=a, b=-0.4 + intercept_noise
            )
        elif constraint_type == 1:
            a = np.zeros(self.dim)
            a[1] = 1.0
            if self.complexity == "simple":
                return loss, AffineFunction(a=a, b=-0.4 + intercept_noise), constraint_type
            H = np.zeros((self.dim, self.dim))
            H[0, 0] = 0.2
            constraint = PSDQuadraticFunction(
                H=H, center=np.zeros(self.dim), a=a, b=-0.4 + intercept_noise
            )
        else:
            a = np.zeros(self.dim)
            a[0] = -1.0
            a[1] = -1.0
            if self.complexity == "simple":
                return loss, AffineFunction(a=a, b=0.85 + intercept_noise), constraint_type
            H = np.zeros((self.dim, self.dim))
            H[0, 0] = 0.2
            H[1, 1] = 0.2
            H[0, 1] = -0.2
            H[1, 0] = -0.2
            constraint = PSDQuadraticFunction(
                H=H, center=np.zeros(self.dim), a=a, b=0.85 + intercept_noise
            )
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
    complexity: str = "simple"

    def __post_init__(self) -> None:
        _validate_dim(self.dim)
        if self.noise_magnitude <= 0:
            raise ValueError("noise_magnitude must be positive")
        _validate_complexity(self.complexity)

    @property
    def feasible_set(self) -> BoxSet:
        return BoxSet(lower=np.zeros(self.dim), upper=np.ones(self.dim))

    @property
    def comparator(self) -> np.ndarray:
        return np.zeros(self.dim)

    def expected_constraint(self, x: np.ndarray) -> float:
        if self.complexity == "simple":
            return float(np.sum(x))
        return float(np.sum(x) + 0.08 * np.dot(x, x))

    def round_wise_feasible_region_nonempty(self, constraint_counts: np.ndarray) -> bool:
        return int(np.asarray(constraint_counts)[1]) == 0

    def sample_round(
        self,
        rng: np.random.Generator,
        round_index: int = 0,
        constraint_rng: np.random.Generator | None = None,
    ) -> tuple[ConvexFunction, ConvexFunction, int]:
        loss = _scheduled_loss(rng, self.dim, round_index, self.complexity)
        constraint_rng = rng if constraint_rng is None else constraint_rng

        noise = _symmetric_uniform_noise(constraint_rng, self.noise_magnitude)
        constraint = _margin_constraint(self.dim, margin=0.0, noise=noise, complexity=self.complexity)
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
    complexity: str = "simple"

    def __post_init__(self) -> None:
        _validate_dim(self.dim)
        if self.margin < 0:
            raise ValueError("margin must be nonnegative")
        if self.noise_magnitude <= 0:
            raise ValueError("noise_magnitude must be positive")
        _validate_complexity(self.complexity)

    @property
    def feasible_set(self) -> BoxSet:
        return BoxSet(lower=np.zeros(self.dim), upper=np.ones(self.dim))

    @property
    def comparator(self) -> np.ndarray:
        boundary_value = (
            self.margin / self.dim
            if self.complexity == "simple"
            else _margin_symmetric_boundary(self.margin, self.dim)
        )
        return np.full(self.dim, boundary_value)

    def expected_constraint(self, x: np.ndarray) -> float:
        if self.complexity == "simple":
            return float(np.sum(x) - self.margin)
        return float(np.sum(x) + 0.08 * np.dot(x, x) - self.margin)

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
        loss = _scheduled_loss(rng, self.dim, round_index, self.complexity)
        constraint_rng = rng if constraint_rng is None else constraint_rng

        noise = _symmetric_uniform_noise(constraint_rng, self.noise_magnitude)
        constraint = _margin_constraint(self.dim, margin=self.margin, noise=noise, complexity=self.complexity)
        return loss, constraint, int(noise > self.margin)


def make_problem(
    name: str,
    dim: int,
    slater_margin: float = 0.0,
    complexity: str = "simple",
) -> ConflictingStochasticConstraints | NoSlaterStochasticConstraints | GradualSlaterStochasticConstraints:
    if name == "slater":
        return ConflictingStochasticConstraints(dim=dim, complexity=complexity)
    if name == "no-slater":
        return NoSlaterStochasticConstraints(dim=dim, complexity=complexity)
    if name == "gradual-slater":
        return GradualSlaterStochasticConstraints(dim=dim, margin=slater_margin, complexity=complexity)
    raise ValueError(f"unknown problem: {name}")


def _validate_complexity(complexity: str) -> None:
    if complexity not in {"simple", "complicated"}:
        raise ValueError("complexity must be either 'simple' or 'complicated'")


def _validate_dim(dim: int) -> None:
    if not MIN_SOLUTION_DIM <= dim <= MAX_SOLUTION_DIM:
        raise ValueError(
            f"dim must be between {MIN_SOLUTION_DIM} and {MAX_SOLUTION_DIM}"
        )


def _scheduled_loss(
    rng: np.random.Generator,
    dim: int,
    round_index: int,
    complexity: str,
) -> ConvexFunction:
    center = _scheduled_center(rng, dim, round_index)
    if complexity == "simple":
        return QuadraticFunction(center=center)
    return _scheduled_composite_loss(rng, dim, round_index, center)


def _scheduled_center(
    rng: np.random.Generator,
    dim: int,
    round_index: int,
) -> np.ndarray:
    # Keep c_t deterministic: every coordinate follows the same 60-round
    # cycle, spending 30 rounds at each endpoint.
    del rng
    t = max(1, int(round_index))
    block = (t - 1) // 30
    base_value = 0.8 if block % 2 == 0 else 0.2
    return np.full(dim, base_value)


def _scheduled_composite_loss(
    rng: np.random.Generator,
    dim: int,
    round_index: int,
    center: np.ndarray | None = None,
) -> ConvexFunction:
    """Generate a nonstationary composite convex loss."""

    t = max(1, int(round_index))
    if center is None:
        center = _scheduled_center(rng, dim, round_index)

    A = np.zeros((4, dim))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 0] = -0.8
    A[2, 1] = 0.6
    A[3, 0] = 0.5
    A[3, 1] = -0.9
    if dim > 2:
        A[:, 2:] = rng.normal(0.0, 0.15, size=(4, dim - 2))

    b = np.array([
        -0.35 + 0.25 * np.sin(2.0 * np.pi * t / 29.0),
        -0.25 + 0.25 * np.cos(2.0 * np.pi * t / 37.0),
        0.10 * np.sin(2.0 * np.pi * t / 17.0),
        0.10 * np.cos(2.0 * np.pi * t / 23.0),
    ])

    kink = np.clip(0.5 + rng.normal(0.0, 0.3, size=dim), 0.0, 1.0)
    linear_tilt = _loss_linear_tilt(rng, dim, round_index)
    return SumFunction(
        [
            QuadraticFunction(center=center),
            LogSumExpAffineFunction(A=A, b=b, weight=0.12),
            SmoothAbsFunction(kink=kink, weight=0.035, epsilon=1e-2),
            AffineFunction(a=linear_tilt),
        ]
    )


def _loss_linear_tilt(
    rng: np.random.Generator,
    dim: int,
    round_index: int,
) -> np.ndarray:
    """Zero-mean x-dependent loss noise with bursts, so regret remains noisy."""

    scale = _time_varying_noise_scale(LOSS_LINEAR_NOISE_SCALE, round_index)
    tilt = rng.normal(0.0, scale, size=dim)
    if rng.random() < LOSS_LINEAR_SHOCK_PROBABILITY:
        tilt += rng.choice([-1.0, 1.0], size=dim) * rng.uniform(
            0.0,
            LOSS_LINEAR_SHOCK_MULTIPLIER * scale,
            size=dim,
        )
    return tilt


def _margin_constraint(dim: int, margin: float, noise: float, complexity: str) -> ConvexFunction:
    a = np.ones(dim)
    if complexity == "simple":
        return AffineFunction(a=a, b=noise - margin)
    H = 0.16 * np.eye(dim)
    return PSDQuadraticFunction(H=H, center=np.zeros(dim), a=a, b=noise - margin)


def _time_varying_noise_scale(base_scale: float, round_index: int) -> float:
    t = max(1, int(round_index))
    smooth_multiplier = 1.0 + 0.35 * np.sin(2.0 * np.pi * t / 127.0)
    burst_multiplier = 1.45 if (t // 70) % 4 == 2 else 1.0
    return float(base_scale * max(0.35, smooth_multiplier) * burst_multiplier)


def _symmetric_uniform_noise(rng: np.random.Generator, magnitude: float) -> float:
    """Sample simple bounded zero-mean noise from Uniform[-magnitude, magnitude]."""

    return float(rng.uniform(-magnitude, magnitude))


def _conflicting_symmetric_boundary() -> float:
    # Solve 0.7*s + 0.09*s^2 - 0.275 = 0.
    return float((-0.7 + np.sqrt(0.7 * 0.7 + 4.0 * 0.09 * 0.275)) / (2.0 * 0.09))


def _margin_symmetric_boundary(margin: float, dim: int) -> float:
    if margin == 0:
        return 0.0
    # Solve dim*s + 0.08*dim*s^2 - margin = 0.
    return float((-dim + np.sqrt(dim * dim + 0.32 * dim * margin)) / (0.16 * dim))
