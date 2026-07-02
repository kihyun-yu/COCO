"""Small convex-function interface used by the algorithms.

The algorithms only require function values and one subgradient at the played
point.  These concrete classes cover common synthetic experiments, while custom
objects can implement the same two methods.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np


class ConvexFunction(Protocol):
    """Protocol for a subdifferentiable convex function on R^d."""

    def value(self, x: np.ndarray) -> float:
        """Return f(x)."""

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Return a subgradient of f at x."""


@dataclass(frozen=True)
class AffineFunction:
    """Affine function f(x) = a^T x + b."""

    a: np.ndarray
    b: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "a", _as_float_vector(self.a, "a"))

    def value(self, x: np.ndarray) -> float:
        return float(np.dot(self.a, _as_float_vector(x, "x")) + self.b)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return self.a.copy()


@dataclass(frozen=True)
class QuadraticFunction:
    """Convex quadratic f(x) = 0.5 ||x - center||_2^2."""

    center: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "center", _as_float_vector(self.center, "center"))

    def value(self, x: np.ndarray) -> float:
        diff = _as_float_vector(x, "x") - self.center
        return float(0.5 * np.dot(diff, diff))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        return _as_float_vector(x, "x") - self.center


@dataclass(frozen=True)
class PSDQuadraticFunction:
    """Convex quadratic f(x) = 0.5 (x-c)^T H (x-c) + a^T x + b."""

    H: np.ndarray
    center: np.ndarray
    a: np.ndarray
    b: float = 0.0

    def __post_init__(self) -> None:
        H = np.array(self.H, dtype=float, copy=True)
        center = _as_float_vector(self.center, "center")
        a = _as_float_vector(self.a, "a")
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("H must be a square matrix")
        if H.shape[0] != center.shape[0] or a.shape != center.shape:
            raise ValueError("H, center, and a dimensions must match")
        if not np.allclose(H, H.T):
            raise ValueError("H must be symmetric")
        if np.min(np.linalg.eigvalsh(H)) < -1e-10:
            raise ValueError("H must be positive semidefinite")
        object.__setattr__(self, "H", H)
        object.__setattr__(self, "center", center)
        object.__setattr__(self, "a", a)

    def value(self, x: np.ndarray) -> float:
        x_vec = _as_float_vector(x, "x")
        diff = x_vec - self.center
        return float(0.5 * diff @ self.H @ diff + np.dot(self.a, x_vec) + self.b)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x_vec = _as_float_vector(x, "x")
        return self.H @ (x_vec - self.center) + self.a


@dataclass(frozen=True)
class LogSumExpAffineFunction:
    """Convex function weight * log(sum_i exp(A_i x + b_i))."""

    A: np.ndarray
    b: np.ndarray
    weight: float = 1.0

    def __post_init__(self) -> None:
        A = np.array(self.A, dtype=float, copy=True)
        b = _as_float_vector(self.b, "b")
        if A.ndim != 2:
            raise ValueError("A must be a matrix")
        if A.shape[0] != b.shape[0]:
            raise ValueError("A rows must match b length")
        if self.weight < 0:
            raise ValueError("weight must be nonnegative")
        object.__setattr__(self, "A", A)
        object.__setattr__(self, "b", b)

    def value(self, x: np.ndarray) -> float:
        z = self.A @ _as_float_vector(x, "x") + self.b
        z_max = float(np.max(z))
        return float(self.weight * (z_max + np.log(np.sum(np.exp(z - z_max)))))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        z = self.A @ _as_float_vector(x, "x") + self.b
        z_shifted = z - np.max(z)
        weights = np.exp(z_shifted)
        weights = weights / np.sum(weights)
        return self.weight * (weights @ self.A)


@dataclass(frozen=True)
class SmoothAbsFunction:
    """Smooth convex approximation of weight * ||x-kink||_1."""

    kink: np.ndarray
    weight: float = 1.0
    epsilon: float = 1e-3

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise ValueError("weight must be nonnegative")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        object.__setattr__(self, "kink", _as_float_vector(self.kink, "kink"))

    def value(self, x: np.ndarray) -> float:
        diff = _as_float_vector(x, "x") - self.kink
        return float(self.weight * np.sum(np.sqrt(diff * diff + self.epsilon * self.epsilon)))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        diff = _as_float_vector(x, "x") - self.kink
        return self.weight * diff / np.sqrt(diff * diff + self.epsilon * self.epsilon)


@dataclass(frozen=True)
class SumFunction:
    """Nonnegative weighted sum of convex functions."""

    terms: Sequence[ConvexFunction]

    def __post_init__(self) -> None:
        if len(self.terms) == 0:
            raise ValueError("terms must be nonempty")

    def value(self, x: np.ndarray) -> float:
        return float(sum(term.value(x) for term in self.terms))

    def gradient(self, x: np.ndarray) -> np.ndarray:
        gradients = [term.gradient(x) for term in self.terms]
        return np.sum(gradients, axis=0)


def _as_float_vector(value: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric NumPy array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array
