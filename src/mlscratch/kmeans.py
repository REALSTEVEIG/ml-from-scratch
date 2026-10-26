"""k-means clustering (Lloyd's algorithm) with k-means++ initialisation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlscratch._validation import (
    RandomState,
    check_array,
    check_is_fitted,
    check_n_features,
    check_non_negative_float,
    check_positive_int,
    check_random_state,
)
from mlscratch.distance import squared_euclidean


class KMeans:
    """Partition data into ``n_clusters`` groups by minimising inertia.

    Inertia is the sum of squared Euclidean distances from each sample to
    its assigned cluster centre.

    Parameters
    ----------
    n_clusters:
        Number of clusters ``k`` (``>= 1``).
    max_iter:
        Maximum number of Lloyd iterations per run.
    tol:
        Convergence threshold: a run stops once no centre moves by more than
        ``tol`` (Euclidean distance) between iterations.
    n_init:
        Number of independent runs with different k-means++ seeds; the run
        with the lowest inertia is kept.
    random_state:
        Seed or ``numpy.random.Generator`` for reproducible results.

    Attributes
    ----------
    cluster_centers_:
        Array of shape ``(n_clusters, n_features)``.
    labels_:
        Cluster index of each training sample, shape ``(n_samples,)``.
    inertia_:
        Inertia of the best run.
    n_iter_:
        Number of iterations performed by the best run.

    Examples
    --------
    >>> X = [[0, 0], [0, 1], [10, 10], [10, 11]]
    >>> km = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> sorted(km.cluster_centers_.tolist())
    [[0.0, 0.5], [10.0, 10.5]]
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 300,
        tol: float = 1e-4,
        n_init: int = 10,
        random_state: RandomState = None,
    ) -> None:
        self.n_clusters = check_positive_int(n_clusters, "n_clusters")
        self.max_iter = check_positive_int(max_iter, "max_iter")
        self.tol = check_non_negative_float(tol, "tol")
        self.n_init = check_positive_int(n_init, "n_init")
        # Validate eagerly so a bad seed fails at construction time.
        check_random_state(random_state)
        self.random_state = random_state

        self.cluster_centers_: NDArray[np.float64] | None = None
        self.labels_: NDArray[np.intp] | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None
        self.n_features_in_: int | None = None

    def __repr__(self) -> str:
        return (
            f"KMeans(n_clusters={self.n_clusters}, max_iter={self.max_iter}, "
            f"tol={self.tol}, n_init={self.n_init}, "
            f"random_state={self.random_state!r})"
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def fit(self, X: ArrayLike) -> KMeans:
        """Compute cluster centres for ``X``.

        Raises
        ------
        ValueError
            If ``X`` is malformed or has fewer samples than ``n_clusters``.
        """
        X_arr = check_array(X)
        if self.n_clusters > X_arr.shape[0]:
            raise ValueError(
                f"n_clusters={self.n_clusters} must be <= the number of "
                f"samples ({X_arr.shape[0]})."
            )
        rng = check_random_state(self.random_state)

        best: tuple[float, NDArray[np.float64], NDArray[np.intp], int] | None = None
        for _ in range(self.n_init):
            centres = self._kmeans_plusplus(X_arr, rng)
            inertia, centres, labels, n_iter = self._lloyd(X_arr, centres)
            # Strict '<' keeps the earliest run on ties -> deterministic.
            if best is None or inertia < best[0]:
                best = (inertia, centres, labels, n_iter)

        assert best is not None
        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.n_features_in_ = X_arr.shape[1]
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Assign each row of ``X`` to its nearest learned cluster centre."""
        check_is_fitted(self, "cluster_centers_")
        assert self.cluster_centers_ is not None and self.n_features_in_ is not None
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)
        return np.argmin(squared_euclidean(X_arr, self.cluster_centers_), axis=1)

    def fit_predict(self, X: ArrayLike) -> NDArray[np.intp]:
        """Fit the model and return the cluster labels of the training data."""
        self.fit(X)
        assert self.labels_ is not None
        return self.labels_

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _kmeans_plusplus(
        self, X: NDArray[np.float64], rng: np.random.Generator
    ) -> NDArray[np.float64]:
        """Choose initial centres with the k-means++ scheme.

        The first centre is a uniformly random sample. Each subsequent centre
        is sampled with probability proportional to its squared distance from
        the nearest centre chosen so far, which spreads centres out and gives
        an ``O(log k)``-competitive solution in expectation.
        """
        n_samples = X.shape[0]
        centres = np.empty((self.n_clusters, X.shape[1]))
        centres[0] = X[rng.integers(n_samples)]
        closest_sq = squared_euclidean(X, centres[:1]).ravel()

        for c in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total == 0.0:
                # All remaining points coincide with existing centres
                # (duplicate data); fall back to a uniform choice.
                idx = int(rng.integers(n_samples))
            else:
                idx = int(rng.choice(n_samples, p=closest_sq / total))
            centres[c] = X[idx]
            new_sq = squared_euclidean(X, centres[c : c + 1]).ravel()
            np.minimum(closest_sq, new_sq, out=closest_sq)
        return centres

    def _lloyd(
        self,
        X: NDArray[np.float64],
        centres: NDArray[np.float64],
    ) -> tuple[float, NDArray[np.float64], NDArray[np.intp], int]:
        """Run Lloyd's algorithm from ``centres`` until convergence.

        Returns
        -------
        inertia, centres, labels, n_iter
        """
        centres = centres.copy()
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            sq_dist = squared_euclidean(X, centres)
            labels = np.argmin(sq_dist, axis=1)
            new_centres = self._update_centres(X, labels, sq_dist)
            shift = np.sqrt(np.sum((new_centres - centres) ** 2, axis=1)).max()
            centres = new_centres
            if shift <= self.tol:
                break

        # Final assignment so labels and inertia match the returned centres.
        sq_dist = squared_euclidean(X, centres)
        labels = np.argmin(sq_dist, axis=1)
        inertia = float(sq_dist[np.arange(X.shape[0]), labels].sum())
        return inertia, centres, labels, n_iter

    def _update_centres(
        self,
        X: NDArray[np.float64],
        labels: NDArray[np.intp],
        sq_dist: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Move each centre to the mean of its assigned samples.

        Empty clusters are re-seeded with the sample that is currently
        farthest from its own centre (ties broken by lowest index), which
        immediately reduces inertia. Each such sample is used at most once.
        """
        counts = np.bincount(labels, minlength=self.n_clusters)
        sums = np.zeros((self.n_clusters, X.shape[1]))
        np.add.at(sums, labels, X)

        new_centres = np.empty_like(sums)
        non_empty = counts > 0
        new_centres[non_empty] = sums[non_empty] / counts[non_empty, np.newaxis]

        empty = np.flatnonzero(~non_empty)
        if empty.size:
            own_dist = sq_dist[np.arange(X.shape[0]), labels]
            # Stable sort on negated distance: farthest first, lowest index on ties.
            farthest = np.argsort(-own_dist, kind="stable")[: empty.size]
            new_centres[empty] = X[farthest]
        return new_centres
