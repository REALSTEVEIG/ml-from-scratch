"""Behavioural tests for KNNClassifier and KNNRegressor."""

from __future__ import annotations

import numpy as np
import pytest
from mlscratch import KNNClassifier, KNNRegressor

# A tiny 1-D dataset that is easy to reason about by hand.
X_LINE = np.array([[0.0], [1.0], [2.0], [10.0], [11.0]])
Y_LINE = np.array(["a", "a", "a", "b", "b"])


def test_hand_checked_majority_vote() -> None:
    clf = KNNClassifier(k=3).fit(X_LINE, Y_LINE)
    # Query 1.5 -> neighbours 1, 2, 0 -> all "a".
    # Query 9.0 -> neighbours 10, 11, 2 -> "b", "b", "a" -> "b".
    assert clf.predict([[1.5], [9.0]]).tolist() == ["a", "b"]


def test_predict_proba_hand_checked() -> None:
    clf = KNNClassifier(k=3).fit(X_LINE, Y_LINE)
    proba = clf.predict_proba([[9.0]])
    assert clf.classes_ is not None
    assert clf.classes_.tolist() == ["a", "b"]
    np.testing.assert_allclose(proba, [[1 / 3, 2 / 3]])


def test_predict_proba_rows_sum_to_one(blobs) -> None:
    X, y = blobs
    proba = KNNClassifier(k=7, weights="distance").fit(X, y).predict_proba(X + 0.3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert proba.shape == (len(X), 3)


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
@pytest.mark.parametrize("weights", ["uniform", "distance"])
def test_blobs_perfect_accuracy(blobs, blobs_test, metric: str, weights: str) -> None:
    X, y = blobs
    X_test, y_test = blobs_test
    clf = KNNClassifier(k=5, metric=metric, weights=weights).fit(X, y)
    assert clf.score(X_test, y_test) == 1.0


@pytest.mark.parametrize("metric", ["euclidean", "manhattan"])
def test_k1_memorises_training_set(metric: str) -> None:
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    y = rng.integers(0, 4, size=40)  # random labels: no structure to learn
    clf = KNNClassifier(k=1, metric=metric).fit(X, y)
    assert clf.score(X, y) == 1.0


def test_k1_regressor_memorises() -> None:
    rng = np.random.default_rng(4)
    X, y = rng.normal(size=(30, 2)), rng.normal(size=30)
    np.testing.assert_allclose(KNNRegressor(k=1).fit(X, y).predict(X), y)


def test_distance_weighting_changes_vote() -> None:
    # Query at 0.0: one very close "a" (d=0.1) and two farther "b"s (d=5).
    X = np.array([[0.1], [5.0], [-5.0]])
    y = np.array(["a", "b", "b"])
    uniform = KNNClassifier(k=3, weights="uniform").fit(X, y)
    weighted = KNNClassifier(k=3, weights="distance").fit(X, y)
    assert uniform.predict([[0.0]]).tolist() == ["b"]
    assert weighted.predict([[0.0]]).tolist() == ["a"]
    # Weights: a = 1/0.1 = 10, b = 2 * 1/5 = 0.4.
    np.testing.assert_allclose(weighted.predict_proba([[0.0]]), [[10 / 10.4, 0.4 / 10.4]])


def test_distance_weighting_exact_match() -> None:
    # A query that coincides with a training point takes that point's label
    # without dividing by zero.
    X = np.array([[0.0], [1.0], [1.1]])
    y = np.array([0, 1, 1])
    clf = KNNClassifier(k=3, weights="distance").fit(X, y)
    with np.errstate(all="raise"):
        np.testing.assert_allclose(clf.predict_proba([[0.0]]), [[1.0, 0.0]])


def test_distance_weighting_multiple_exact_matches() -> None:
    X = np.array([[0.0], [0.0], [0.0], [5.0]])
    y = np.array([0, 1, 1, 0])
    clf = KNNClassifier(k=4, weights="distance").fit(X, y)
    np.testing.assert_allclose(clf.predict_proba([[0.0]]), [[1 / 3, 2 / 3]])


def test_vote_tie_goes_to_smallest_label() -> None:
    X = np.array([[-1.0], [1.0]])
    clf = KNNClassifier(k=2).fit(X, np.array(["z", "y"]))
    # Equal votes for "y" and "z" -> sorted classes_ = ["y", "z"] -> "y".
    assert clf.predict([[0.0]]).tolist() == ["y"]


def test_distance_tie_goes_to_earliest_training_point() -> None:
    # Points at -1 and +1 are equidistant from 0; k=1 must pick index 0.
    X = np.array([[1.0], [-1.0]])
    clf = KNNClassifier(k=1).fit(X, np.array([7, 3]))
    _, idx = clf.kneighbors([[0.0]])
    assert idx.tolist() == [[0]]
    assert clf.predict([[0.0]]).tolist() == [7]


def test_kneighbors_sorted_nearest_first() -> None:
    clf = KNNClassifier(k=3).fit(X_LINE, Y_LINE)
    dist, idx = clf.kneighbors([[10.4]])
    assert idx.tolist() == [[3, 4, 2]]
    np.testing.assert_allclose(dist, [[0.4, 0.6, 8.4]])


def test_manhattan_differs_from_euclidean() -> None:
    # From the origin: A=(3,3) has L2 4.24 / L1 6; B=(0,5) has L2 5 / L1 5.
    X = np.array([[3.0, 3.0], [0.0, 5.0]])
    y = np.array(["A", "B"])
    assert KNNClassifier(k=1).fit(X, y).predict([[0, 0]]).tolist() == ["A"]
    assert KNNClassifier(k=1, metric="manhattan").fit(X, y).predict([[0, 0]]).tolist() == ["B"]


def test_integer_and_string_labels_preserved() -> None:
    X = np.array([[0.0], [10.0]])
    assert KNNClassifier(k=1).fit(X, [5, 9]).predict([[1.0]]).dtype.kind == "i"
    assert KNNClassifier(k=1).fit(X, ["no", "yes"]).predict([[9.0]]).tolist() == ["yes"]


def test_fit_returns_self_and_accepts_lists() -> None:
    clf = KNNClassifier(k=1)
    assert clf.fit([[0, 0], [1, 1]], [0, 1]) is clf
    assert clf.n_features_in_ == 2


def test_repr() -> None:
    assert repr(KNNClassifier(k=3, metric="manhattan")) == (
        "KNNClassifier(k=3, metric='manhattan', weights='uniform')"
    )


# --------------------------------------------------------------------------- #
# Regressor
# --------------------------------------------------------------------------- #


def test_regressor_uniform_mean() -> None:
    reg = KNNRegressor(k=2).fit([[0.0], [1.0], [10.0]], [0.0, 2.0, 50.0])
    np.testing.assert_allclose(reg.predict([[0.4], [9.0]]), [1.0, 26.0])


def test_regressor_distance_weighted() -> None:
    reg = KNNRegressor(k=2, weights="distance").fit([[0.0], [3.0]], [0.0, 3.0])
    # Query 1: distances 1 and 2 -> weights 1 and 0.5 -> (0*1 + 3*0.5) / 1.5 = 1.
    np.testing.assert_allclose(reg.predict([[1.0]]), [1.0])


def test_regressor_score_r2() -> None:
    X = np.linspace(0, 1, 20)[:, None]
    y = 3 * X.ravel()
    assert KNNRegressor(k=1).fit(X, y).score(X, y) == pytest.approx(1.0)
    # Predicting a constant (k = n) gives R^2 == 0 on the training data.
    assert KNNRegressor(k=20).fit(X, y).score(X, y) == pytest.approx(0.0)


def test_regressor_score_constant_target() -> None:
    reg = KNNRegressor(k=1).fit([[0.0], [1.0]], [2.0, 2.0])
    assert reg.score([[0.0], [1.0]], [2.0, 2.0]) == 1.0
    assert reg.score([[0.0], [1.0]], [3.0, 3.0]) == 0.0
