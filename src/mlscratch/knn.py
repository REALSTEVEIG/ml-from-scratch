"""k-nearest-neighbour classifier and regressor."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from mlscratch._validation import (
    check_array,
    check_is_fitted,
    check_n_features,
    check_option,
    check_positive_int,
    check_target,
)
from mlscratch.distance import METRICS, Metric, _pairwise_validated

Weights = Literal["uniform", "distance"]
WEIGHTS: tuple[str, ...] = ("uniform", "distance")


class _KNNBase:
    """Shared logic for kNN models: parameter checks, storage and neighbour search.

    kNN is a *lazy* learner: ``fit`` only validates and stores the training
    data; all real work happens at prediction time.
    """

    _numeric_target: bool = False

    def __init__(
        self,
        k: int = 5,
        metric: Metric = "euclidean",
        weights: Weights = "uniform",
    ) -> None:
        self.k = check_positive_int(k, "k")
        self.metric = check_option(metric, "metric", METRICS)
        self.weights = check_option(weights, "weights", WEIGHTS)
        self.X_train_: NDArray[np.float64] | None = None
        self.y_train_: NDArray[Any] | None = None
        self.n_features_in_: int | None = None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(k={self.k}, metric={self.metric!r}, "
            f"weights={self.weights!r})"
        )

    def fit(self, X: ArrayLike, y: ArrayLike) -> _KNNBase:
        """Store the training data.

        Parameters
        ----------
        X:
            Training features, shape ``(n_samples, n_features)``.
        y:
            Training targets, shape ``(n_samples,)``.

        Returns
        -------
        self
            The fitted model, to allow method chaining.

        Raises
        ------
        ValueError
            If the inputs are malformed or ``k`` exceeds ``n_samples``.
        """
        X_arr = check_array(X)
        y_arr = check_target(y, X_arr.shape[0], numeric=self._numeric_target)
        if self.k > X_arr.shape[0]:
            raise ValueError(
                f"k={self.k} must be <= the number of training samples "
                f"({X_arr.shape[0]})."
            )
        self.X_train_ = X_arr
        self.y_train_ = y_arr
        self.n_features_in_ = X_arr.shape[1]
        return self

    def kneighbors(
        self, X: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
        """Find the ``k`` nearest training points for each query row.

        Ties in distance are broken deterministically in favour of the
        training point that appears first in the training set (a stable
        sort is used).

        Returns
        -------
        distances, indices
            Both of shape ``(n_queries, k)``, ordered nearest first.
        """
        check_is_fitted(self, "X_train_")
        assert self.X_train_ is not None and self.n_features_in_ is not None
        X_arr = check_array(X)
        check_n_features(X_arr, self.n_features_in_)

        dist = _pairwise_validated(X_arr, self.X_train_, self.metric)
        idx = np.argsort(dist, axis=1, kind="stable")[:, : self.k]
        return np.take_along_axis(dist, idx, axis=1), idx

    def _neighbour_weights(self, dist: NDArray[np.float64]) -> NDArray[np.float64]:
        """Weight for each neighbour, shape ``(n_queries, k)``.

        With ``weights="distance"`` the weight is ``1 / d``. If a query
        coincides exactly with one or more training points (``d == 0``),
        those points receive weight 1 and all others weight 0, avoiding
        division by zero.
        """
        if self.weights == "uniform":
            return np.ones_like(dist)
        zero = dist == 0.0
        with np.errstate(divide="ignore"):
            w = 1.0 / dist
        exact_rows = zero.any(axis=1)
        w[exact_rows] = zero[exact_rows].astype(np.float64)
        return w


class KNNClassifier(_KNNBase):
    """k-nearest-neighbour classifier.

    Parameters
    ----------
    k:
        Number of neighbours that vote (``>= 1``).
    metric:
        ``"euclidean"`` or ``"manhattan"``.
    weights:
        ``"uniform"`` (each neighbour gets one vote) or ``"distance"``
        (votes weighted by inverse distance).

    Attributes
    ----------
    classes_:
        Sorted array of the distinct training labels.

    Notes
    -----
    If two classes receive the same total vote, the class that sorts first
    in ``classes_`` wins, so predictions are fully deterministic.

    Examples
    --------
    >>> clf = KNNClassifier(k=1).fit([[0], [10]], ["a", "b"])
    >>> clf.predict([[2], [9]]).tolist()
    ['a', 'b']
    """

    def __init__(
        self,
        k: int = 5,
        metric: Metric = "euclidean",
        weights: Weights = "uniform",
    ) -> None:
        super().__init__(k=k, metric=metric, weights=weights)
        self.classes_: NDArray[Any] | None = None
        self._y_encoded: NDArray[np.intp] | None = None

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNClassifier:
        """Store the training data and the set of class labels."""
        super().fit(X, y)
        self.classes_, self._y_encoded = np.unique(self.y_train_, return_inverse=True)
        return self

    def predict_proba(self, X: ArrayLike) -> NDArray[np.float64]:
        """Estimate class membership probabilities.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_queries, n_classes)``; column ``j`` corresponds to
            ``classes_[j]`` and each row sums to 1.
        """
        dist, idx = self.kneighbors(X)
        assert self.classes_ is not None and self._y_encoded is not None
        w = self._neighbour_weights(dist)
        neighbour_classes = self._y_encoded[idx]  # (n_queries, k)

        votes = np.zeros((idx.shape[0], len(self.classes_)))
        rows = np.repeat(np.arange(idx.shape[0]), idx.shape[1])
        np.add.at(votes, (rows, neighbour_classes.ravel()), w.ravel())
        return votes / votes.sum(axis=1, keepdims=True)

    def predict(self, X: ArrayLike) -> NDArray[Any]:
        """Predict the class label for each query row."""
        proba = self.predict_proba(X)
        assert self.classes_ is not None
        # np.argmax returns the first maximum -> ties go to the smallest label.
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return classification accuracy on ``(X, y)``."""
        y_pred = self.predict(X)
        y_true = check_target(y, len(y_pred), numeric=False)
        return float(np.mean(y_pred == y_true))


class KNNRegressor(_KNNBase):
    """k-nearest-neighbour regressor.

    The prediction is the (optionally distance-weighted) mean of the
    targets of the ``k`` nearest training points.

    Examples
    --------
    >>> reg = KNNRegressor(k=2).fit([[0], [1], [10]], [0.0, 2.0, 50.0])
    >>> reg.predict([[0.4]]).tolist()
    [1.0]
    """

    _numeric_target = True

    def fit(self, X: ArrayLike, y: ArrayLike) -> KNNRegressor:
        """Store the training data (targets must be finite numbers)."""
        super().fit(X, y)
        return self

    def predict(self, X: ArrayLike) -> NDArray[np.float64]:
        """Predict a continuous target for each query row."""
        dist, idx = self.kneighbors(X)
        assert self.y_train_ is not None
        w = self._neighbour_weights(dist)
        return np.sum(w * self.y_train_[idx], axis=1) / np.sum(w, axis=1)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return the coefficient of determination R^2 on ``(X, y)``.

        If ``y`` is constant, returns 1.0 for a perfect fit and 0.0 otherwise.
        """
        y_pred = self.predict(X)
        y_true = check_target(y, len(y_pred), numeric=True)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
        if ss_tot == 0.0:
            return 1.0 if ss_res == 0.0 else 0.0
        return 1.0 - ss_res / ss_tot
