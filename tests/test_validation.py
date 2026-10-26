"""Tests for the shared validation helpers, covering every error path."""

from __future__ import annotations

import numpy as np
import pytest
from mlscratch import KMeans, KNNClassifier, KNNRegressor, NotFittedError
from mlscratch._validation import (
    check_array,
    check_non_negative_float,
    check_positive_int,
    check_random_state,
    check_target,
)

# --------------------------------------------------------------------------- #
# check_array
# --------------------------------------------------------------------------- #


def test_check_array_converts_to_float64() -> None:
    out = check_array([[1, 2], [3, 4]])
    assert out.dtype == np.float64
    assert out.shape == (2, 2)


def test_check_array_accepts_bool() -> None:
    assert check_array([[True, False]]).tolist() == [[1.0, 0.0]]


@pytest.mark.parametrize(
    "bad, msg",
    [
        ([1.0, 2.0], "2-D array"),
        ([[[1.0]]], "2-D array"),
        (5.0, "2-D array"),
        (np.empty((0, 3)), "at least one sample"),
        (np.empty((3, 0)), "at least one sample"),
        ([[1.0, np.nan]], "NaN or infinite"),
        ([[1.0, np.inf]], "NaN or infinite"),
        ([[1.0, -np.inf]], "NaN or infinite"),
        ([[1, 2], [3]], "could not be converted|2-D array"),
    ],
)
def test_check_array_value_errors(bad: object, msg: str) -> None:
    with pytest.raises(ValueError, match=msg):
        check_array(bad)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [[["a", "b"]], [[1 + 2j]], np.array([[None]], dtype=object)])
def test_check_array_type_errors(bad: object) -> None:
    with pytest.raises(TypeError, match="numeric"):
        check_array(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# check_target
# --------------------------------------------------------------------------- #


def test_check_target_not_1d() -> None:
    with pytest.raises(ValueError, match="1-D"):
        check_target([[0], [1]], 2, numeric=False)


def test_check_target_length_mismatch() -> None:
    with pytest.raises(ValueError, match="inconsistent lengths"):
        check_target([0, 1, 2], 2, numeric=False)


def test_check_target_numeric_type_error() -> None:
    with pytest.raises(TypeError, match="numeric"):
        check_target(["a", "b"], 2, numeric=True)


@pytest.mark.parametrize("numeric", [True, False])
def test_check_target_nan(numeric: bool) -> None:
    with pytest.raises(ValueError, match="NaN or infinite"):
        check_target([0.0, np.nan], 2, numeric=numeric)


def test_check_target_allows_string_labels() -> None:
    assert check_target(["a", "b"], 2, numeric=False).tolist() == ["a", "b"]


# --------------------------------------------------------------------------- #
# Scalar parameter checks
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [1.5, "3", None, True])
def test_check_positive_int_type(bad: object) -> None:
    with pytest.raises(TypeError, match="must be an integer"):
        check_positive_int(bad, "k")


@pytest.mark.parametrize("bad", [0, -1])
def test_check_positive_int_value(bad: int) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        check_positive_int(bad, "k")


def test_check_positive_int_accepts_numpy_int() -> None:
    assert check_positive_int(np.int64(3), "k") == 3


@pytest.mark.parametrize("bad", ["0.1", None, False])
def test_check_non_negative_float_type(bad: object) -> None:
    with pytest.raises(TypeError, match="real number"):
        check_non_negative_float(bad, "tol")


@pytest.mark.parametrize("bad", [-1e-3, np.nan, np.inf])
def test_check_non_negative_float_value(bad: float) -> None:
    with pytest.raises(ValueError, match="finite number >= 0"):
        check_non_negative_float(bad, "tol")


def test_check_random_state_variants() -> None:
    g = np.random.default_rng(0)
    assert check_random_state(g) is g
    assert isinstance(check_random_state(None), np.random.Generator)
    a = check_random_state(7).random()
    b = check_random_state(7).random()
    assert a == b


@pytest.mark.parametrize("bad", ["seed", 1.5, True])
def test_check_random_state_type(bad: object) -> None:
    with pytest.raises(TypeError, match="random_state"):
        check_random_state(bad)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Estimator-level error paths
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", [KNNClassifier, KNNRegressor])
def test_knn_not_fitted(cls: type) -> None:
    model = cls(k=1)
    with pytest.raises(NotFittedError, match="not fitted yet"):
        model.predict([[0.0]])
    with pytest.raises(NotFittedError):
        model.kneighbors([[0.0]])


def test_knn_classifier_not_fitted_proba_and_score() -> None:
    clf = KNNClassifier(k=1)
    with pytest.raises(NotFittedError):
        clf.predict_proba([[0.0]])
    with pytest.raises(NotFittedError):
        clf.score([[0.0]], [0])


def test_kmeans_not_fitted() -> None:
    with pytest.raises(NotFittedError, match="KMeans instance is not fitted"):
        KMeans(n_clusters=2).predict([[0.0]])


def test_not_fitted_error_is_runtime_error() -> None:
    assert issubclass(NotFittedError, RuntimeError)


@pytest.mark.parametrize(
    "kwargs, exc, msg",
    [
        ({"k": 0}, ValueError, "k must be >= 1"),
        ({"k": 2.0}, TypeError, "k must be an integer"),
        ({"metric": "cosine"}, ValueError, "metric must be one of"),
        ({"weights": "gaussian"}, ValueError, "weights must be one of"),
    ],
)
def test_knn_bad_params(kwargs: dict, exc: type, msg: str) -> None:
    with pytest.raises(exc, match=msg):
        KNNClassifier(**kwargs)


def test_knn_k_larger_than_n_samples() -> None:
    with pytest.raises(ValueError, match="k=3 must be <= the number of training samples"):
        KNNClassifier(k=3).fit([[0.0], [1.0]], [0, 1])


def test_knn_fit_length_mismatch() -> None:
    with pytest.raises(ValueError, match="inconsistent lengths"):
        KNNClassifier(k=1).fit([[0.0], [1.0]], [0])


def test_knn_fit_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        KNNClassifier(k=1).fit([[0.0], [np.nan]], [0, 1])


def test_knn_regressor_rejects_string_targets() -> None:
    with pytest.raises(TypeError, match="y must contain numeric"):
        KNNRegressor(k=1).fit([[0.0], [1.0]], ["a", "b"])


def test_knn_predict_feature_mismatch() -> None:
    clf = KNNClassifier(k=1).fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])
    with pytest.raises(ValueError, match="X has 3 features, but the model was fitted with 2"):
        clf.predict([[0.0, 0.0, 0.0]])


def test_knn_predict_1d_input() -> None:
    clf = KNNClassifier(k=1).fit([[0.0, 0.0], [1.0, 1.0]], [0, 1])
    with pytest.raises(ValueError, match="2-D"):
        clf.predict([0.0, 0.0])


def test_knn_score_length_mismatch() -> None:
    clf = KNNClassifier(k=1).fit([[0.0], [1.0]], [0, 1])
    with pytest.raises(ValueError, match="inconsistent lengths"):
        clf.score([[0.0], [1.0]], [0])


@pytest.mark.parametrize(
    "kwargs, exc, msg",
    [
        ({"n_clusters": 0}, ValueError, "n_clusters must be >= 1"),
        ({"n_clusters": 2.5}, TypeError, "n_clusters must be an integer"),
        ({"max_iter": 0}, ValueError, "max_iter must be >= 1"),
        ({"n_init": 0}, ValueError, "n_init must be >= 1"),
        ({"tol": -1.0}, ValueError, "tol must be a finite number"),
        ({"tol": "small"}, TypeError, "tol must be a real number"),
        ({"random_state": "abc"}, TypeError, "random_state"),
    ],
)
def test_kmeans_bad_params(kwargs: dict, exc: type, msg: str) -> None:
    with pytest.raises(exc, match=msg):
        KMeans(**kwargs)


def test_kmeans_too_many_clusters() -> None:
    with pytest.raises(ValueError, match="n_clusters=3 must be <= the number of samples"):
        KMeans(n_clusters=3).fit([[0.0], [1.0]])


def test_kmeans_fit_bad_input() -> None:
    with pytest.raises(ValueError, match="NaN"):
        KMeans(n_clusters=1).fit([[np.inf]])
    with pytest.raises(TypeError, match="numeric"):
        KMeans(n_clusters=1).fit([["x"]])


def test_kmeans_predict_feature_mismatch() -> None:
    km = KMeans(n_clusters=1, random_state=0).fit([[0.0, 0.0], [1.0, 1.0]])
    with pytest.raises(ValueError, match="X has 1 features, but the model was fitted with 2"):
        km.predict([[0.0]])
