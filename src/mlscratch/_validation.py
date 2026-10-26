"""Shared input-validation helpers and the custom ``NotFittedError``.

Every public estimator funnels its inputs through these functions so that
error messages are consistent and each check lives in exactly one place.
"""

from __future__ import annotations

import numbers
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

# NumPy dtype kinds accepted as numeric input: bool, signed int, unsigned
# int and float. Complex, string and object arrays are rejected.
_NUMERIC_KINDS = frozenset("biuf")

RandomState = int | np.random.Generator | None


class NotFittedError(RuntimeError):
    """Raised when ``predict``-style methods are called before ``fit``."""


def check_array(X: ArrayLike, name: str = "X") -> NDArray[np.float64]:
    """Validate a 2-D numeric feature matrix and return it as ``float64``.

    Parameters
    ----------
    X:
        Array-like of shape ``(n_samples, n_features)``.
    name:
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        A ``float64`` copy-or-view of ``X``.

    Raises
    ------
    TypeError
        If ``X`` is not numeric.
    ValueError
        If ``X`` is not 2-D, is empty, or contains NaN / infinite values.
    """
    try:
        arr = np.asarray(X)
    except (ValueError, TypeError) as exc:  # e.g. ragged nested lists
        raise ValueError(f"{name} could not be converted to an array: {exc}") from exc

    if arr.dtype.kind not in _NUMERIC_KINDS:
        raise TypeError(
            f"{name} must contain numeric values, got dtype {arr.dtype!s}."
        )
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (n_samples, n_features), "
            f"got an array with {arr.ndim} dimension(s)."
        )
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(
            f"{name} must have at least one sample and one feature, "
            f"got shape {arr.shape}."
        )
    arr = arr.astype(np.float64, copy=False)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def check_target(y: ArrayLike, n_samples: int, *, numeric: bool) -> NDArray[Any]:
    """Validate a 1-D target vector whose length must equal ``n_samples``.

    Parameters
    ----------
    y:
        Array-like of shape ``(n_samples,)``.
    n_samples:
        Expected number of entries (the number of rows in ``X``).
    numeric:
        If ``True`` the targets must be finite numbers (regression);
        otherwise any hashable, sortable labels are allowed (classification).

    Raises
    ------
    TypeError
        If ``numeric`` is ``True`` and ``y`` is not numeric.
    ValueError
        If ``y`` is not 1-D, has the wrong length, or contains NaN / inf.
    """
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(
            f"y must be a 1-D array, got an array with {arr.ndim} dimension(s)."
        )
    if arr.shape[0] != n_samples:
        raise ValueError(
            f"X and y have inconsistent lengths: X has {n_samples} samples, "
            f"y has {arr.shape[0]}."
        )
    if numeric:
        if arr.dtype.kind not in _NUMERIC_KINDS:
            raise TypeError(f"y must contain numeric values, got dtype {arr.dtype!s}.")
        arr = arr.astype(np.float64, copy=False)
        if not np.all(np.isfinite(arr)):
            raise ValueError("y contains NaN or infinite values.")
    elif arr.dtype.kind == "f" and not np.all(np.isfinite(arr)):
        raise ValueError("y contains NaN or infinite values.")
    return arr


def check_positive_int(value: Any, name: str) -> int:
    """Return ``value`` as ``int`` if it is a positive integer.

    Booleans are rejected even though ``bool`` subclasses ``int``.

    Raises
    ------
    TypeError
        If ``value`` is not an integer.
    ValueError
        If ``value`` is smaller than 1.
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}.")
    return int(value)


def check_non_negative_float(value: Any, name: str) -> float:
    """Return ``value`` as ``float`` if it is a finite, non-negative real."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}.")
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value}.")
    return float(value)


def check_option(value: Any, name: str, options: tuple[str, ...]) -> str:
    """Ensure ``value`` is one of the allowed string ``options``."""
    if value not in options:
        raise ValueError(f"{name} must be one of {options}, got {value!r}.")
    return str(value)


def check_n_features(X: NDArray[Any], n_features: int) -> None:
    """Raise ``ValueError`` if ``X`` does not have ``n_features`` columns."""
    if X.shape[1] != n_features:
        raise ValueError(
            f"X has {X.shape[1]} features, but the model was fitted with "
            f"{n_features} features."
        )


def check_is_fitted(estimator: object, attribute: str) -> None:
    """Raise :class:`NotFittedError` if ``estimator`` lacks ``attribute``."""
    if getattr(estimator, attribute, None) is None:
        raise NotFittedError(
            f"This {type(estimator).__name__} instance is not fitted yet. "
            "Call 'fit' with appropriate data before using this method."
        )


def check_random_state(seed: RandomState) -> np.random.Generator:
    """Turn ``seed`` into a :class:`numpy.random.Generator`.

    ``None`` gives fresh OS entropy, an ``int`` gives a reproducible
    generator, and an existing ``Generator`` is returned unchanged.
    """
    if seed is None or (
        isinstance(seed, numbers.Integral) and not isinstance(seed, bool)
    ):
        return np.random.default_rng(seed)
    if isinstance(seed, np.random.Generator):
        return seed
    raise TypeError(
        "random_state must be None, an int or a numpy.random.Generator, "
        f"got {type(seed).__name__}."
    )
