"""Behavioural tests for KMeans."""

from __future__ import annotations

import numpy as np
import pytest
from mlscratch import KMeans


def _same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True if label vectors ``a`` and ``b`` define the same grouping."""
    pairs = set(zip(a.tolist(), b.tolist()))
    return len(pairs) == len(set(a.tolist())) == len(set(b.tolist()))


def test_hand_checked_two_clusters() -> None:
    X = np.array([[0.0, 0.0], [0.0, 2.0], [10.0, 10.0], [10.0, 12.0]])
    km = KMeans(n_clusters=2, random_state=0).fit(X)
    assert km.cluster_centers_ is not None and km.labels_ is not None
    assert sorted(km.cluster_centers_.tolist()) == [[0.0, 1.0], [10.0, 11.0]]
    # Each point is 1 unit from its centre -> inertia = 4 * 1^2.
    assert km.inertia_ == pytest.approx(4.0)
    assert km.labels_[0] == km.labels_[1] != km.labels_[2] == km.labels_[3]


def test_recovers_blobs(blobs, blob_centres) -> None:
    X, y = blobs
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.labels_ is not None and km.cluster_centers_ is not None
    assert _same_partition(km.labels_, y)
    # Each learned centre is close to one true centre.
    order = np.argsort(km.cluster_centers_[:, 0] + 2 * km.cluster_centers_[:, 1])
    np.testing.assert_allclose(km.cluster_centers_[order], blob_centres, atol=0.5)


def test_predict_matches_labels_and_new_points(blobs, blobs_test) -> None:
    X, _ = blobs
    X_test, y_test = blobs_test
    km = KMeans(n_clusters=3, random_state=1).fit(X)
    np.testing.assert_array_equal(km.predict(X), km.labels_)
    assert _same_partition(km.predict(X_test), y_test)


def test_fit_predict_equals_fit_then_labels(blobs) -> None:
    X, _ = blobs
    labels = KMeans(n_clusters=3, random_state=5).fit_predict(X)
    km = KMeans(n_clusters=3, random_state=5).fit(X)
    np.testing.assert_array_equal(labels, km.labels_)


def test_reproducible_with_random_state(blobs) -> None:
    X, _ = blobs
    X = X + np.random.default_rng(9).normal(scale=4.0, size=X.shape)  # harder data
    a = KMeans(n_clusters=4, n_init=3, random_state=123).fit(X)
    b = KMeans(n_clusters=4, n_init=3, random_state=123).fit(X)
    np.testing.assert_array_equal(a.cluster_centers_, b.cluster_centers_)
    np.testing.assert_array_equal(a.labels_, b.labels_)
    assert a.inertia_ == b.inertia_
    assert a.n_iter_ == b.n_iter_


def test_accepts_generator_random_state(blobs) -> None:
    X, _ = blobs
    km = KMeans(n_clusters=3, random_state=np.random.default_rng(0)).fit(X)
    assert km.labels_ is not None and _same_partition(km.labels_, blobs[1])


def test_more_restarts_never_worse() -> None:
    X = np.random.default_rng(2).normal(size=(200, 2))
    single = KMeans(n_clusters=6, n_init=1, random_state=0).fit(X)
    many = KMeans(n_clusters=6, n_init=10, random_state=0).fit(X)
    # The first restart of `many` uses the same seed stream as `single`,
    # and the best of 10 can only be as good or better.
    assert many.inertia_ <= single.inertia_ + 1e-12


def test_inertia_matches_definition(blobs) -> None:
    X, _ = blobs
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.cluster_centers_ is not None and km.labels_ is not None
    expected = np.sum((X - km.cluster_centers_[km.labels_]) ** 2)
    assert km.inertia_ == pytest.approx(expected)


def test_centres_are_cluster_means(blobs) -> None:
    X, _ = blobs
    km = KMeans(n_clusters=3, random_state=0, tol=0.0).fit(X)
    assert km.cluster_centers_ is not None and km.labels_ is not None
    for j in range(3):
        np.testing.assert_allclose(km.cluster_centers_[j], X[km.labels_ == j].mean(axis=0))


def test_converges_quickly_on_separated_data(blobs) -> None:
    X, _ = blobs
    km = KMeans(n_clusters=3, max_iter=300, random_state=0).fit(X)
    assert km.n_iter_ is not None and 1 <= km.n_iter_ < 10


def test_max_iter_caps_iterations() -> None:
    X = np.random.default_rng(0).normal(size=(500, 2))
    km = KMeans(n_clusters=10, max_iter=2, tol=0.0, n_init=1, random_state=0).fit(X)
    assert km.n_iter_ == 2


def test_tol_zero_reaches_fixed_point() -> None:
    X = np.random.default_rng(1).normal(size=(100, 2))
    km = KMeans(n_clusters=4, tol=0.0, n_init=1, random_state=0).fit(X)
    assert km.cluster_centers_ is not None
    # One more Lloyd step must not move any centre.
    labels = km.predict(X)
    moved = np.array([X[labels == j].mean(axis=0) for j in range(4)])
    np.testing.assert_allclose(moved, km.cluster_centers_)


def test_single_cluster_is_global_mean() -> None:
    X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
    km = KMeans(n_clusters=1, random_state=0).fit(X)
    np.testing.assert_allclose(km.cluster_centers_, [[3.0, 5.0]])
    assert km.labels_ is not None and km.labels_.tolist() == [0, 0, 0]


def test_k_equals_n_gives_zero_inertia() -> None:
    X = np.array([[0.0], [1.0], [5.0], [9.0]])
    km = KMeans(n_clusters=4, random_state=0).fit(X)
    assert km.inertia_ == pytest.approx(0.0)
    assert km.cluster_centers_ is not None
    assert sorted(km.cluster_centers_.ravel().tolist()) == [0.0, 1.0, 5.0, 9.0]


def test_duplicate_points_with_more_clusters_than_distinct_values() -> None:
    # Only 2 distinct points but 3 clusters: k-means++ must not divide by zero
    # and empty clusters must be handled without producing NaN centres.
    X = np.array([[0.0, 0.0]] * 4 + [[5.0, 5.0]] * 4)
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.cluster_centers_ is not None
    assert np.all(np.isfinite(km.cluster_centers_))
    assert km.inertia_ == pytest.approx(0.0)


def test_all_identical_points() -> None:
    X = np.ones((5, 2))
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.cluster_centers_ is not None
    np.testing.assert_allclose(km.cluster_centers_, 1.0)
    assert km.inertia_ == 0.0


def test_empty_cluster_is_reseeded_with_farthest_point() -> None:
    # Direct test of the update step: centre 2 starts far from all data,
    # so it receives no points and must be moved to the point farthest
    # from its current centre (x=4, which is 3 away from centre 1).
    X = np.array([[0.0], [1.0], [4.0]])
    km = KMeans(n_clusters=3, n_init=1, random_state=0)
    centres = np.array([[0.0], [1.0], [100.0]])
    sq_dist = (X - centres.T) ** 2
    labels = np.argmin(sq_dist, axis=1)
    assert 2 not in labels  # cluster 2 really is empty
    new = km._update_centres(X, labels, sq_dist)
    np.testing.assert_allclose(new, [[0.0], [2.5], [4.0]])


def test_lloyd_recovers_from_empty_cluster() -> None:
    X = np.array([[0.0], [0.5], [10.0], [10.5]])
    km = KMeans(n_clusters=2, n_init=1, random_state=0)
    inertia, centres, labels, _ = km._lloyd(X, np.array([[5.0], [1000.0]]))
    assert sorted(centres.ravel().tolist()) == [0.25, 10.25]
    assert inertia == pytest.approx(4 * 0.25**2)
    assert labels[0] == labels[1] != labels[2] == labels[3]


def test_kmeans_plusplus_picks_distinct_spread_points() -> None:
    X = np.array([[0.0, 0.0], [0.1, 0.0], [50.0, 50.0], [-50.0, 50.0]])
    km = KMeans(n_clusters=3, random_state=0)
    centres = km._kmeans_plusplus(X, np.random.default_rng(0))
    # Every chosen centre is an actual data point and all are distinct.
    assert all(any(np.array_equal(c, x) for x in X) for c in centres)
    assert len({tuple(c) for c in centres}) == 3


def test_repr() -> None:
    assert repr(KMeans(n_clusters=2, random_state=1)) == (
        "KMeans(n_clusters=2, max_iter=300, tol=0.0001, n_init=10, random_state=1)"
    )
