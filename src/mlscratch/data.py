"""Small data utilities: CSV loading, train/test splitting, scaling and blobs."""

from __future__ import annotations

import csv
import numbers
import os
from dataclasses import dataclass, field
from typing import Any

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


class DataError(ValueError):
    """Raised when a data file or data argument is malformed."""


@dataclass
class Dataset:
    """A loaded table: numeric features plus an optional label column.

    Attributes
    ----------
    X:
        ``float64`` feature matrix of shape ``(n_samples, n_features)``.
    y:
        Labels of shape ``(n_samples,)`` as strings, or ``None`` if the
        file was loaded without a label column.
    feature_names:
        One name per feature column.
    label_name:
        Name of the label column, or ``None``.
    """

    X: NDArray[np.float64]
    y: NDArray[np.str_] | None
    feature_names: list[str] = field(default_factory=list)
    label_name: str | None = None


def load_csv(
    path: str | os.PathLike[str],
    label_column: str | int | None = -1,
    *,
    header: bool = True,
    delimiter: str = ",",
) -> Dataset:
    """Load a CSV file of numeric features and an optional label column.

    Parameters
    ----------
    path:
        Path to the CSV file.
    label_column:
        Which column holds the labels: a column index (negative values count
        from the end, default ``-1`` = last column), a header name (requires
        ``header=True``), or ``None`` if every column is a feature.
    header:
        Whether the first row contains column names.
    delimiter:
        Field separator.

    Returns
    -------
    Dataset

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    DataError
        If the file is empty, ragged, has non-numeric feature values,
        missing values, or ``label_column`` cannot be resolved.
    """
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise DataError(f"delimiter must be a single character, got {delimiter!r}.")
    if label_column is not None and (
        isinstance(label_column, bool) or not isinstance(label_column, (str, numbers.Integral))
    ):
        raise TypeError(
            "label_column must be a column name, an integer index or None, "
            f"got {type(label_column).__name__}."
        )

    try:
        with open(path, newline="", encoding="utf-8") as fh:
            rows = [row for row in csv.reader(fh, delimiter=delimiter) if row]
    except UnicodeDecodeError as exc:
        raise DataError(f"{path}: not a UTF-8 text file ({exc.reason}).") from exc
    except csv.Error as exc:
        raise DataError(f"{path}: invalid CSV ({exc}).") from exc

    if not rows:
        raise DataError(f"{path}: file is empty.")

    if header:
        names = [name.strip() for name in rows[0]]
        body = rows[1:]
        line_offset = 2
    else:
        names = [f"x{i}" for i in range(len(rows[0]))]
        body = rows
        line_offset = 1
    if not body:
        raise DataError(f"{path}: file has a header but no data rows.")

    n_cols = len(names)
    for i, row in enumerate(body):
        if len(row) != n_cols:
            raise DataError(
                f"{path}, line {i + line_offset}: expected {n_cols} fields, got {len(row)}."
            )

    label_idx = _resolve_label_column(label_column, names, header=header, path=path)
    feature_idx = [j for j in range(n_cols) if j != label_idx]
    if not feature_idx:
        raise DataError(f"{path}: no feature columns left after removing the label column.")

    X = np.empty((len(body), len(feature_idx)), dtype=np.float64)
    for i, row in enumerate(body):
        for out_j, j in enumerate(feature_idx):
            cell = row[j].strip()
            try:
                X[i, out_j] = float(cell)
            except ValueError:
                raise DataError(
                    f"{path}, line {i + line_offset}, column {names[j]!r}: "
                    f"expected a number, got {cell!r}."
                ) from None
    if not np.all(np.isfinite(X)):
        raise DataError(f"{path}: feature columns contain NaN or infinite values.")

    y: NDArray[np.str_] | None = None
    if label_idx is not None:
        labels = [row[label_idx].strip() for row in body]
        if any(label == "" for label in labels):
            raise DataError(f"{path}: label column {names[label_idx]!r} has missing values.")
        y = np.array(labels, dtype=str)

    return Dataset(
        X=X,
        y=y,
        feature_names=[names[j] for j in feature_idx],
        label_name=None if label_idx is None else names[label_idx],
    )


def _resolve_label_column(
    label_column: str | int | None,
    names: list[str],
    *,
    header: bool,
    path: str | os.PathLike[str],
) -> int | None:
    """Turn a column name / index into a non-negative index (or ``None``)."""
    if label_column is None:
        return None
    n_cols = len(names)
    if isinstance(label_column, str):
        if not header:
            raise DataError("label_column can only be a name when header=True.")
        if label_column not in names:
            raise DataError(
                f"{path}: label column {label_column!r} not found; columns are {names}."
            )
        return names.index(label_column)
    idx = int(label_column)
    if not -n_cols <= idx < n_cols:
        raise DataError(
            f"{path}: label column index {idx} is out of range for {n_cols} columns."
        )
    return idx % n_cols


def train_test_split(
    X: ArrayLike,
    y: ArrayLike | None = None,
    *,
    test_size: float | int = 0.25,
    random_state: RandomState = None,
    stratify: bool = False,
) -> tuple[NDArray[Any], ...]:
    """Shuffle and split arrays into train and test parts.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, n_features)``.
    y:
        Optional targets of shape ``(n_samples,)``.
    test_size:
        Fraction of samples in ``(0, 1)`` or an absolute count
        ``1 <= test_size < n_samples``.
    random_state:
        Seed or generator for a reproducible shuffle.
    stratify:
        If ``True`` (requires ``y``), keep each class's share roughly equal
        in both parts.

    Returns
    -------
    tuple
        ``(X_train, X_test)`` or ``(X_train, X_test, y_train, y_test)``.
    """
    X_arr = check_array(X)
    n = X_arr.shape[0]
    y_arr: NDArray[Any] | None = None
    if y is not None:
        y_arr = np.asarray(y)
        if y_arr.ndim != 1 or y_arr.shape[0] != n:
            raise ValueError(f"y must be 1-D with {n} entries, got shape {y_arr.shape}.")
    if stratify and y_arr is None:
        raise ValueError("stratify=True requires y.")

    n_test = _resolve_test_size(test_size, n)
    rng = check_random_state(random_state)

    if stratify:
        assert y_arr is not None
        test_idx = _stratified_test_indices(y_arr, n_test, rng)
    else:
        test_idx = rng.permutation(n)[:n_test]
    mask = np.zeros(n, dtype=bool)
    mask[test_idx] = True
    train_idx = rng.permutation(np.flatnonzero(~mask))
    test_idx = rng.permutation(np.flatnonzero(mask))

    if y_arr is None:
        return X_arr[train_idx], X_arr[test_idx]
    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def _resolve_test_size(test_size: float | int, n: int) -> int:
    if isinstance(test_size, bool) or not isinstance(test_size, numbers.Real):
        raise TypeError(f"test_size must be a number, got {type(test_size).__name__}.")
    if isinstance(test_size, numbers.Integral):
        n_test = int(test_size)
    else:
        if not 0.0 < test_size < 1.0:
            raise ValueError(f"A fractional test_size must be in (0, 1), got {test_size}.")
        n_test = int(np.ceil(test_size * n))
    if not 1 <= n_test < n:
        raise ValueError(
            f"test_size={test_size} gives {n_test} test samples; need between 1 and "
            f"{n - 1} for {n} samples."
        )
    return n_test


def _stratified_test_indices(
    y: NDArray[Any], n_test: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Pick ``n_test`` indices with class shares matching ``y`` (largest remainder)."""
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    exact = counts * n_test / len(y)
    per_class = np.floor(exact).astype(int)
    leftover = n_test - per_class.sum()
    # Hand out the remaining slots to classes with the largest remainders.
    order = np.argsort(-(exact - per_class), kind="stable")
    per_class[order[:leftover]] += 1
    chosen = [
        rng.permutation(np.flatnonzero(inverse == c))[: per_class[c]]
        for c in range(len(classes))
    ]
    return np.concatenate(chosen)


class StandardScaler:
    """Rescale each feature to zero mean and unit variance.

    Constant features (zero variance) are only centred, not divided.
    """

    def __init__(self) -> None:
        self.mean_: NDArray[np.float64] | None = None
        self.scale_: NDArray[np.float64] | None = None

    def fit(self, X: ArrayLike) -> StandardScaler:
        """Learn the per-feature mean and standard deviation of ``X``."""
        X_arr = check_array(X)
        self.mean_ = X_arr.mean(axis=0)
        std = X_arr.std(axis=0)
        self.scale_ = np.where(std > 0.0, std, 1.0)
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Apply the learned scaling to ``X``."""
        check_is_fitted(self, "mean_")
        assert self.mean_ is not None and self.scale_ is not None
        X_arr = check_array(X)
        check_n_features(X_arr, len(self.mean_))
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X: ArrayLike) -> NDArray[np.float64]:
        """Fit to ``X`` and return the scaled ``X``."""
        return self.fit(X).transform(X)


def make_blobs(
    n_samples: int = 100,
    centers: int | ArrayLike = 3,
    *,
    n_features: int = 2,
    cluster_std: float = 1.0,
    center_box: tuple[float, float] = (-10.0, 10.0),
    random_state: RandomState = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Generate isotropic Gaussian blobs for clustering / classification.

    Parameters
    ----------
    n_samples:
        Total number of points, split as evenly as possible across clusters
        (earlier clusters get the extra points).
    centers:
        Number of clusters (centres drawn uniformly from ``center_box``) or
        an explicit array of shape ``(n_centers, n_features)``.
    n_features:
        Dimensionality; ignored when ``centers`` is an array.
    cluster_std:
        Standard deviation of every blob.
    center_box:
        ``(low, high)`` bounds for random centres.
    random_state:
        Seed or generator.

    Returns
    -------
    X, y
        Points of shape ``(n_samples, n_features)`` and their blob index.
    """
    n_samples = check_positive_int(n_samples, "n_samples")
    cluster_std = check_non_negative_float(cluster_std, "cluster_std")
    rng = check_random_state(random_state)

    if isinstance(centers, numbers.Integral) and not isinstance(centers, bool):
        n_centers = check_positive_int(centers, "centers")
        n_features = check_positive_int(n_features, "n_features")
        low, high = (float(v) for v in center_box)
        if not (np.isfinite(low) and np.isfinite(high) and low < high):
            raise ValueError(f"center_box must be finite with low < high, got {center_box}.")
        centre_arr = rng.uniform(low, high, size=(n_centers, n_features))
    else:
        centre_arr = check_array(centers, name="centers")
        n_centers = centre_arr.shape[0]
    if n_centers > n_samples:
        raise ValueError(f"Need n_samples >= number of centers ({n_samples} < {n_centers}).")

    sizes = np.full(n_centers, n_samples // n_centers)
    sizes[: n_samples % n_centers] += 1
    y = np.repeat(np.arange(n_centers), sizes)
    X = centre_arr[y] + cluster_std * rng.standard_normal((n_samples, centre_arr.shape[1]))
    return X, y
