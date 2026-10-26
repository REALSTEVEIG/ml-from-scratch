"""Tests for the data utilities: load_csv, train_test_split, StandardScaler, make_blobs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mlscratch import (
    DataError,
    NotFittedError,
    StandardScaler,
    load_csv,
    make_blobs,
    train_test_split,
)

IRIS = Path(__file__).resolve().parents[1] / "data" / "iris.csv"


def write(tmp_path: Path, text: str, name: str = "d.csv") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------------------- #
# Bundled Iris dataset
# --------------------------------------------------------------------------- #
def test_iris_shape_and_balance() -> None:
    ds = load_csv(IRIS)
    assert ds.X.shape == (150, 4)
    assert ds.feature_names == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert ds.label_name == "species"
    assert ds.y is not None
    classes, counts = np.unique(ds.y, return_counts=True)
    assert classes.tolist() == ["setosa", "versicolor", "virginica"]
    assert counts.tolist() == [50, 50, 50]


def test_iris_known_values() -> None:
    """Spot-check against Fisher (1936), incl. the rows the UCI copy gets wrong."""
    ds = load_csv(IRIS)
    np.testing.assert_allclose(ds.X[0], [5.1, 3.5, 1.4, 0.2])
    np.testing.assert_allclose(ds.X[34], [4.9, 3.1, 1.5, 0.2])
    np.testing.assert_allclose(ds.X[37], [4.9, 3.6, 1.4, 0.1])
    np.testing.assert_allclose(ds.X[-1], [5.9, 3.0, 5.1, 1.8])
    # Well-known column means.
    np.testing.assert_allclose(ds.X.mean(axis=0), [5.8433, 3.0573, 3.7580, 1.1993], atol=1e-4)


# --------------------------------------------------------------------------- #
# load_csv
# --------------------------------------------------------------------------- #
def test_load_csv_label_by_name_and_index(tmp_path: Path) -> None:
    p = write(tmp_path, "cls,a,b\nx,1,2\ny,3,4\n")
    by_name = load_csv(p, "cls")
    by_index = load_csv(p, 0)
    for ds in (by_name, by_index):
        np.testing.assert_array_equal(ds.X, [[1, 2], [3, 4]])
        assert ds.y is not None and ds.y.tolist() == ["x", "y"]
        assert ds.feature_names == ["a", "b"]
        assert ds.label_name == "cls"


def test_load_csv_no_label_no_header_and_delimiter(tmp_path: Path) -> None:
    p = write(tmp_path, "1;2\n3;4\n\n")
    ds = load_csv(p, None, header=False, delimiter=";")
    np.testing.assert_array_equal(ds.X, [[1, 2], [3, 4]])
    assert ds.y is None and ds.label_name is None
    assert ds.feature_names == ["x0", "x1"]


def test_load_csv_numeric_labels_kept_as_strings(tmp_path: Path) -> None:
    ds = load_csv(write(tmp_path, "a,y\n1.5,0\n2.5,1\n"))
    assert ds.y is not None and ds.y.tolist() == ["0", "1"]


def test_load_csv_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")


@pytest.mark.parametrize(
    ("text", "kwargs", "match"),
    [
        ("", {}, "empty"),
        ("a,b,y\n", {}, "no data rows"),
        ("a,b,y\n1,2,x\n3,x\n", {}, "line 3: expected 3 fields"),
        ("a,b,y\n1,oops,x\n", {}, "column 'b': expected a number"),
        ("a,b,y\n1,,x\n", {}, "expected a number"),
        ("a,b,y\n1,nan,x\n", {}, "NaN or infinite"),
        ("a,b,y\n1,2,\n", {}, "missing values"),
        ("a,b,y\n1,2,x\n", {"label_column": "zzz"}, "not found"),
        ("a,b,y\n1,2,x\n", {"label_column": 3}, "out of range"),
        ("y\nx\n", {}, "no feature columns"),
        ("1,2\n", {"label_column": "a", "header": False}, "header=True"),
        ("a,b\n1,2\n", {"delimiter": ";;"}, "single character"),
    ],
)
def test_load_csv_errors(tmp_path: Path, text: str, kwargs: dict, match: str) -> None:
    with pytest.raises(DataError, match=match):
        load_csv(write(tmp_path, text), **kwargs)


def test_load_csv_binary_file(tmp_path: Path) -> None:
    p = tmp_path / "bin.csv"
    p.write_bytes(b"\xff\xfe\x00\x81")
    with pytest.raises(DataError, match="UTF-8"):
        load_csv(p)


def test_load_csv_bad_label_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="label_column"):
        load_csv(write(tmp_path, "a,y\n1,x\n"), label_column=1.5)  # type: ignore[arg-type]


def test_data_error_is_value_error() -> None:
    assert issubclass(DataError, ValueError)


# --------------------------------------------------------------------------- #
# train_test_split
# --------------------------------------------------------------------------- #
def test_split_sizes_and_partition() -> None:
    X = np.arange(20, dtype=float).reshape(10, 2)
    y = np.arange(10)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
    assert len(X_tr) == 7 and len(X_te) == 3
    assert sorted(np.concatenate([y_tr, y_te]).tolist()) == list(range(10))
    # Rows stay aligned with their labels.
    np.testing.assert_array_equal(X_tr[:, 0], 2 * y_tr)
    np.testing.assert_array_equal(X_te[:, 0], 2 * y_te)


def test_split_is_reproducible_and_seed_dependent() -> None:
    X = np.arange(100, dtype=float).reshape(50, 2)
    a = train_test_split(X, random_state=1)
    b = train_test_split(X, random_state=1)
    c = train_test_split(X, random_state=2)
    np.testing.assert_array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])
    assert len(a) == 2


def test_split_absolute_test_size() -> None:
    X = np.zeros((10, 1))
    X_tr, X_te = train_test_split(X, test_size=4, random_state=0)
    assert (len(X_tr), len(X_te)) == (6, 4)


def test_split_stratified_preserves_class_shares() -> None:
    y = np.array(["a"] * 60 + ["b"] * 30 + ["c"] * 10)
    X = np.zeros((100, 1))
    _, _, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=0, stratify=True)
    assert dict(zip(*np.unique(y_te, return_counts=True))) == {"a": 12, "b": 6, "c": 2}
    assert len(y_tr) == 80


def test_split_stratified_largest_remainder() -> None:
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    _, _, _, y_te = train_test_split(np.zeros((9, 1)), y, test_size=4, random_state=0, stratify=True)
    assert len(y_te) == 4
    assert sorted(np.bincount(y_te, minlength=3).tolist()) == [1, 1, 2]


@pytest.mark.parametrize(
    ("kwargs", "exc", "match"),
    [
        ({"test_size": 0.0}, ValueError, r"\(0, 1\)"),
        ({"test_size": 1.0}, ValueError, r"\(0, 1\)"),
        ({"test_size": 0}, ValueError, "between 1 and"),
        ({"test_size": 10}, ValueError, "between 1 and"),
        ({"test_size": "0.2"}, TypeError, "number"),
        ({"test_size": True}, TypeError, "number"),
        ({"stratify": True}, ValueError, "requires y"),
        ({"random_state": "x"}, TypeError, "random_state"),
    ],
)
def test_split_errors(kwargs: dict, exc: type[Exception], match: str) -> None:
    with pytest.raises(exc, match=match):
        train_test_split(np.zeros((10, 2)), **kwargs)


def test_split_y_length_mismatch() -> None:
    with pytest.raises(ValueError, match="10 entries"):
        train_test_split(np.zeros((10, 2)), np.zeros(9))


# --------------------------------------------------------------------------- #
# StandardScaler
# --------------------------------------------------------------------------- #
def test_scaler_zero_mean_unit_variance_and_constant_column() -> None:
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z = StandardScaler().fit_transform(X)
    np.testing.assert_allclose(Z.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(Z[:, 0].std(), 1.0)
    np.testing.assert_array_equal(Z[:, 1], 0.0)


def test_scaler_errors() -> None:
    with pytest.raises(NotFittedError):
        StandardScaler().transform([[1.0]])
    sc = StandardScaler().fit([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="features"):
        sc.transform([[1.0]])


# --------------------------------------------------------------------------- #
# make_blobs
# --------------------------------------------------------------------------- #
def test_make_blobs_shapes_and_balance() -> None:
    X, y = make_blobs(n_samples=10, centers=3, n_features=4, random_state=0)
    assert X.shape == (10, 4)
    assert np.bincount(y).tolist() == [4, 3, 3]


def test_make_blobs_reproducible() -> None:
    a = make_blobs(50, 2, random_state=7)
    b = make_blobs(50, 2, random_state=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_make_blobs_explicit_centres_and_zero_std() -> None:
    centres = np.array([[0.0, 0.0], [10.0, 10.0]])
    X, y = make_blobs(4, centres, cluster_std=0.0, random_state=0)
    np.testing.assert_array_equal(X, centres[y])


def test_make_blobs_points_near_centres() -> None:
    centres = np.array([[0.0, 0.0], [50.0, 0.0]])
    X, y = make_blobs(400, centres, cluster_std=1.0, random_state=0)
    for c in range(2):
        np.testing.assert_allclose(X[y == c].mean(axis=0), centres[c], atol=0.3)


@pytest.mark.parametrize(
    ("args", "kwargs", "exc"),
    [
        ((0,), {}, ValueError),
        ((5, 0), {}, ValueError),
        ((2, 3), {}, ValueError),
        ((10,), {"cluster_std": -1.0}, ValueError),
        ((10,), {"center_box": (1.0, 1.0)}, ValueError),
        ((10,), {"n_features": 0}, ValueError),
        ((10, [[np.nan, 0.0]]), {}, ValueError),
        ((10,), {"random_state": 1.5}, TypeError),
    ],
)
def test_make_blobs_errors(args: tuple, kwargs: dict, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        make_blobs(*args, **kwargs)
