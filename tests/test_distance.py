"""Tests for pairwise distance computations."""

from __future__ import annotations

import numpy as np
import pytest
from mlscratch import pairwise_distances


def test_euclidean_hand_checked() -> None:
    D = pairwise_distances([[0, 0], [1, 1]], [[3, 4], [1, 1]])
    np.testing.assert_allclose(D, [[5.0, np.sqrt(2)], [np.sqrt(13), 0.0]])


def test_manhattan_hand_checked() -> None:
    D = pairwise_distances([[0, 0], [1, 1]], [[3, 4], [1, 1]], metric="manhattan")
    np.testing.assert_allclose(D, [[7.0, 2.0], [5.0, 0.0]])


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_matches_naive_loop(metric: str) -> None:
    rng = np.random.default_rng(42)
    A, B = rng.normal(size=(7, 4)), rng.normal(size=(5, 4))
    order = 2 if metric == "euclidean" else 1
    expected = np.array([[np.linalg.norm(a - b, ord=order) for b in B] for a in A])
    np.testing.assert_allclose(pairwise_distances(A, B, metric=metric), expected)


def test_self_distance_non_negative_and_near_zero() -> None:
    # The dot-product expansion suffers cancellation error of roughly
    # sqrt(eps) * ||x||, so the diagonal is only ~0 at large scales; the
    # clamp must still guarantee no negative (or NaN) distances.
    A = np.random.default_rng(0).normal(size=(20, 3)) * 1e3
    D = pairwise_distances(A, A)
    assert np.all(D >= 0)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-3)


def test_invalid_metric() -> None:
    with pytest.raises(ValueError, match="metric must be one of"):
        pairwise_distances([[0]], [[1]], metric="cosine")  # type: ignore[arg-type]


def test_feature_mismatch() -> None:
    with pytest.raises(ValueError, match="same number of features"):
        pairwise_distances([[0, 0]], [[1, 1, 1]])
