"""Algorithms from the two COCO papers in ./papers.

Implemented papers:
- Yu, Neely, Wei (2017), Algorithm 1: drift-plus-penalty with virtual queues.
- Yu, Lee, Lee (2026), Algorithm 1: anytime primal-dual COCO without Slater.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Sequence

import numpy as np

from functions import ConvexFunction
from sets import ConvexSet


@dataclass(frozen=True)
class YuNeelyWeiConfig:
    """Parameters for Yu-Neely-Wei 2017 Algorithm 1.

    The paper uses fixed constants V and alpha.  For horizon-dependent theory,
    common choices are V=sqrt(T) and alpha=T, but this implementation keeps them
    explicit so experiments can tune them.
    """

    V: float
    alpha: float

    def __post_init__(self) -> None:
        if self.V <= 0:
            raise ValueError("V must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")


@dataclass
class YuNeelyWei2017:
    """Yu, Neely, Wei (2017) OCO with stochastic constraints.

    Supports multiple scalar convex constraints g_k.  A call to ``step`` plays
    the current decision x_t, observes f_t and g_t, then updates x_{t+1} and the
    virtual queues Q_{t+1}.
    """

    feasible_set: ConvexSet
    x: np.ndarray
    num_constraints: int
    config: YuNeelyWeiConfig
    Q: np.ndarray = field(init=False)
    t: int = field(default=1, init=False)

    def __post_init__(self) -> None:
        if self.num_constraints <= 0:
            raise ValueError("num_constraints must be positive")
        self.x = self.feasible_set.project(_as_float_vector(self.x, "x"))
        self.Q = np.zeros(self.num_constraints, dtype=float)

    def step(
        self, loss: ConvexFunction, constraints: Sequence[ConvexFunction]
    ) -> dict[str, np.ndarray | float | int]:
        if len(constraints) != self.num_constraints:
            raise ValueError("constraints length must match num_constraints")

        x_t = self.x.copy()
        loss_value = loss.value(x_t)
        constraint_values = np.array([g.value(x_t) for g in constraints], dtype=float)

        direction = self.config.V * _as_float_vector(loss.gradient(x_t), "loss gradient")
        constraint_gradients = []
        for q_k, g_k in zip(self.Q, constraints):
            grad = _as_float_vector(g_k.gradient(x_t), "constraint gradient")
            constraint_gradients.append(grad)
            direction = direction + q_k * grad

        x_next = self.feasible_set.project(x_t - direction / (2.0 * self.config.alpha))

        for k, (g_value, grad) in enumerate(zip(constraint_values, constraint_gradients)):
            linearized = g_value + float(np.dot(grad, x_next - x_t))
            self.Q[k] = max(self.Q[k] + linearized, 0.0)

        self.x = x_next
        self.t += 1
        return {
            "t": self.t - 1,
            "x": x_t,
            "loss": loss_value,
            "constraints": constraint_values,
            "queue": self.Q.copy(),
            "x_next": x_next.copy(),
        }


@dataclass
class YuNeelyWei2017Doubling:
    """Anytime wrapper for Yu-Neely-Wei 2017 using the doubling trick.

    The base 2017 algorithm needs a horizon-dependent parameter choice.  This
    wrapper runs it in epochs of length 1, 2, 4, ... and restarts the virtual
    queues/parameters at each epoch while warm-starting the primal decision from
    the previous epoch's last point.
    """

    feasible_set: ConvexSet
    x: np.ndarray
    num_constraints: int
    epoch_horizon: int = field(default=1, init=False)
    epoch_round: int = field(default=0, init=False)
    t: int = field(default=1, init=False)
    current: YuNeelyWei2017 = field(init=False)

    def __post_init__(self) -> None:
        if self.num_constraints <= 0:
            raise ValueError("num_constraints must be positive")
        self.x = self.feasible_set.project(_as_float_vector(self.x, "x"))
        self.current = self._new_epoch_algorithm(self.x, self.epoch_horizon)

    def step(
        self, loss: ConvexFunction, constraints: Sequence[ConvexFunction]
    ) -> dict[str, np.ndarray | float | int]:
        if self.epoch_round >= self.epoch_horizon:
            self.epoch_horizon *= 2
            self.epoch_round = 0
            self.current = self._new_epoch_algorithm(self.current.x, self.epoch_horizon)

        out = self.current.step(loss, constraints)
        self.x = self.current.x.copy()
        self.epoch_round += 1
        self.t += 1
        out["t"] = self.t - 1
        out["epoch_horizon"] = self.epoch_horizon
        out["epoch_round"] = self.epoch_round
        return out

    def _new_epoch_algorithm(self, x: np.ndarray, horizon: int) -> YuNeelyWei2017:
        return YuNeelyWei2017(
            feasible_set=self.feasible_set,
            x=x,
            num_constraints=self.num_constraints,
            config=YuNeelyWeiConfig(V=math.sqrt(horizon), alpha=float(horizon)),
        )


@dataclass(frozen=True)
class COCOWithoutSlaterConfig:
    """Parameters for Yu-Lee-Lee 2026 Algorithm 1.

    This paper states Algorithm 1 for a scalar constraint g_t.  Set
    ``high_probability=True`` to use the paper's delta-dependent Psi and gamma_t.
    Set ``mu>0`` to use the strongly-convex primal step size.
    """

    L: float
    G: float
    D: float
    delta: float = 0.05
    mu: float = 0.0
    high_probability: bool = False
    gamma_scale: float = 1.0
    regularizer_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.L <= 0 or self.G <= 0 or self.D <= 0:
            raise ValueError("L, G, and D must be positive")
        if not 0.0 < self.delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        if self.mu < 0:
            raise ValueError("mu must be nonnegative")
        if self.gamma_scale <= 0:
            raise ValueError("gamma_scale must be positive")
        if self.regularizer_scale < 0:
            raise ValueError("regularizer_scale must be nonnegative")


@dataclass
class COCOWithoutSlater2026:
    """Yu, Lee, Lee (2026) anytime primal-dual algorithm without Slater.

    The implementation follows Algorithm 1:
    x_{t+1}=proj_X(x_t-eta_t(grad f_t(x_t)+Phi'_t(Q_t) grad g_t(x_t)))
    Q_{t+1}=Q_t+g_t(x_t)-R_t.
    """

    feasible_set: ConvexSet
    x: np.ndarray
    config: COCOWithoutSlaterConfig
    Q: float = field(default=0.0, init=False)
    t: int = field(default=1, init=False)
    surrogate_norm_sq_sum: float = field(default=0.0, init=False)
    phi_prime_sq_sum: float = field(default=0.0, init=False)

    def __post_init__(self) -> None:
        self.x = self.feasible_set.project(_as_float_vector(self.x, "x"))

    def step(self, loss: ConvexFunction, constraint: ConvexFunction) -> dict[str, np.ndarray | float | int]:
        x_t = self.x.copy()
        gamma_t = self._gamma(self.t)
        phi_prime = gamma_t * math.exp(gamma_t * self.Q)
        loss_grad = _as_float_vector(loss.gradient(x_t), "loss gradient")
        constraint_grad = _as_float_vector(constraint.gradient(x_t), "constraint gradient")
        surrogate_grad = loss_grad + phi_prime * constraint_grad

        self.surrogate_norm_sq_sum += float(np.dot(surrogate_grad, surrogate_grad))
        eta_t = self._eta()
        x_next = self.feasible_set.project(x_t - eta_t * surrogate_grad)

        previous_phi_sum = self.phi_prime_sq_sum
        self.phi_prime_sq_sum += phi_prime * phi_prime
        regularizer = (self._psi(self.phi_prime_sq_sum) - self._psi(previous_phi_sum)) / phi_prime

        loss_value = loss.value(x_t)
        constraint_value = constraint.value(x_t)
        self.Q = self.Q + constraint_value - self.config.regularizer_scale * regularizer
        self.x = x_next
        self.t += 1
        return {
            "t": self.t - 1,
            "x": x_t,
            "loss": loss_value,
            "constraint": float(constraint_value),
            "Q": float(self.Q),
            "gamma": float(gamma_t),
            "phi_prime": float(phi_prime),
            "eta": float(eta_t),
            "regularizer": float(regularizer),
            "x_next": x_next.copy(),
        }

    def _eta(self) -> float:
        if self.config.mu > 0.0:
            denom = (
                self.config.mu * self.t
                + (2.0 * self.config.L / self.config.D) * math.sqrt(self.phi_prime_sq_sum)
            )
            return 1.0 / max(denom, 1e-12)
        return math.sqrt(2.0) * self.config.D / (
            2.0 * math.sqrt(1.0 + self.surrogate_norm_sq_sum)
        )

    def _psi(self, x: float) -> float:
        base = 4.0 * self.config.D * self.config.L * math.sqrt(max(x, 0.0))
        if not self.config.high_probability:
            return base

        one_plus_x = 1.0 + max(x, 0.0)
        log2_term = math.log(one_plus_x, 2.0)
        log_arg = ((math.pi * math.pi / 6.0) * (1.0 + log2_term) ** 2) / self.config.delta
        return base + 4.0 * self.config.G * math.sqrt(one_plus_x * math.log(log_arg))

    def _gamma(self, t: int) -> float:
        if self.config.high_probability:
            confidence_term = math.sqrt(math.log(12.0 * t * t / self.config.delta))
            base_gamma = min(
                1.0 / (12.0 * self.config.G * math.sqrt(t)),
                1.0 / (24.0 * (self.config.D * self.config.L + 8.0 * self.config.G * confidence_term)),
                1.0,
            )
            return self.config.gamma_scale * base_gamma
        base_gamma = min(
            1.0 / (12.0 * self.config.G * math.sqrt(t)),
            1.0 / (24.0 * self.config.D * self.config.L),
            1.0,
        )
        return self.config.gamma_scale * base_gamma


def _as_float_vector(value: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric NumPy array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array
