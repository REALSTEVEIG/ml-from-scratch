"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray


def make_blobs(
    centres: NDArray[np.float64], n_per: int, spread: float, seed: int
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Generate isotropic Gaussian blobs around ``centres``."""
    rng = np.random.default_rng(seed)
    X = np.vstack([c + spread * rng.standard_normal((n_per, len(c))) for c in centres])
    y = np.repeat(np.arange(len(centres)), n_per)
    return X, y


@pytest.fixture
def blob_centres() -> NDArray[np.float64]:
    return np.array([[0.0, 0.0], [20.0, 0.0], [0.0, 20.0]])


@pytest.fixture
def blobs(blob_centres: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Three well-separated 2-D blobs, 50 points each."""
    return make_blobs(blob_centres, n_per=50, spread=1.0, seed=0)


@pytest.fixture
def blobs_test(
    blob_centres: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Held-out samples from the same three blobs."""
    return make_blobs(blob_centres, n_per=20, spread=1.0, seed=1)
