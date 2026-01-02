"""Potential functions for 1D systems."""
from typing import Optional

import numpy as np


def V_isw(x: np.ndarray, L: Optional[float] = None) -> np.ndarray:
    """Infinite square well potential (zero inside the domain).

    Args:
        x: 1D array of grid positions.
        L: Optional domain length (unused, kept for API symmetry).

    Returns:
        Array of zeros with the same shape as x.
    """
    return np.zeros_like(x, dtype=float)


def V_ho(x: np.ndarray, omega: float, m: float = 1.0) -> np.ndarray:
    """Harmonic oscillator potential: 0.5 * m * omega^2 * x^2.

    Args:
        x: 1D array of grid positions.
        omega: Oscillator frequency.
        m: Mass (default 1.0).

    Returns:
        Potential values evaluated at each grid point.
    """
    return 0.5 * m * (omega**2) * x**2


def V_doublewell(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Double well potential: a * x^4 - b * x^2.

    Args:
        x: 1D array of grid positions.
        a: Quartic coefficient.
        b: Quadratic coefficient.

    Returns:
        Potential values evaluated at each grid point.
    """
    return a * x**4 - b * x**2


__all__ = ["V_isw", "V_ho", "V_doublewell"]
