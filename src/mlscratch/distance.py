"""Vectorised pairwise distance computations."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlscratch._validation import check_array, check_option

Metric = Literal["euclidean", "manhattan"]
METRICS: tuple[str, ...] = ("euclidean", "manhattan")


def squared_euclidean(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    """Squared Euclidean distances between the rows of ``A`` and ``B``.

    Uses the identity ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` so the
    heavy lifting is a single matrix multiplication, needing only
    ``O(m * n)`` extra memory instead of ``O(m * n * d)``.

    Inputs are assumed to be validated ``float64`` arrays with matching
    numbers of columns.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(len(A), len(B))``.
    """
    a_sq = np.einsum("ij,ij->i", A, A)[:, np.newaxis]
    b_sq = np.einsum("ij,ij->i", B, B)[np.newaxis, :]
    dist = a_sq + b_sq - 2.0 * (A @ B.T)
    # Floating-point cancellation can produce tiny negative values.
    np.maximum(dist, 0.0, out=dist)
    return dist


def _manhattan(A: NDArray[np.float64], B: NDArray[np.float64]) -> NDArray[np.float64]:
    """Manhattan (L1) distances via broadcasting; uses ``O(m * n * d)`` memory."""
    return np.abs(A[:, np.newaxis, :] - B[np.newaxis, :, :]).sum(axis=2)


def pairwise_distances(
    A: ArrayLike, B: ArrayLike, metric: Metric = "euclidean"
) -> NDArray[np.float64]:
    """Compute the distance between every row of ``A`` and every row of ``B``.

    Parameters
    ----------
    A:
        Array-like of shape ``(m, d)``.
    B:
        Array-like of shape ``(n, d)``.
    metric:
        ``"euclidean"`` (L2) or ``"manhattan"`` (L1).

    Returns
    -------
    numpy.ndarray
        Matrix ``D`` of shape ``(m, n)`` with ``D[i, j] = dist(A[i], B[j])``.

    Examples
    --------
    >>> pairwise_distances([[0, 0]], [[3, 4], [1, 1]])
    array([[5.        , 1.41421356]])
    """
    check_option(metric, "metric", METRICS)
    A_arr = check_array(A, name="A")
    B_arr = check_array(B, name="B")
    if A_arr.shape[1] != B_arr.shape[1]:
        raise ValueError(
            f"A and B must have the same number of features, got "
            f"{A_arr.shape[1]} and {B_arr.shape[1]}."
        )
    return _pairwise_validated(A_arr, B_arr, metric)


def _pairwise_validated(
    A: NDArray[np.float64], B: NDArray[np.float64], metric: str
) -> NDArray[np.float64]:
    """Distance matrix for inputs that have already been validated."""
    if metric == "euclidean":
        return np.sqrt(squared_euclidean(A, B))
    return _manhattan(A, B)
