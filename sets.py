"""Projection sets for constrained online convex optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class ConvexSet(Protocol):
    """Known simple compact convex set X used by projected updates."""

    def project(self, x: np.ndarray) -> np.ndarray:
        """Return the Euclidean projection of x onto the set."""

    @property
    def diameter(self) -> float:
        """Return an upper bound on ||x - y|| for x, y in the set."""


@dataclass(frozen=True)
class BoxSet:
    """Axis-aligned box {x: lower <= x <= upper}."""

    lower: np.ndarray
    upper: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", _as_float_vector(self.lower, "lower"))
        object.__setattr__(self, "upper", _as_float_vector(self.upper, "upper"))
        if self.lower.shape != self.upper.shape:
            raise ValueError("lower and upper must have the same shape")
        if np.any(self.lower > self.upper):
            raise ValueError("lower must be <= upper coordinate-wise")

    def project(self, x: np.ndarray) -> np.ndarray:
        return np.clip(_as_float_vector(x, "x"), self.lower, self.upper)

    @property
    def diameter(self) -> float:
        return float(np.linalg.norm(self.upper - self.lower))


def _as_float_vector(value: np.ndarray, name: str) -> np.ndarray:
    try:
        array = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a numeric NumPy array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array
