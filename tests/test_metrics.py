"""Tests for accuracy_score, confusion_matrix, cluster_purity and formatting."""

from __future__ import annotations

import numpy as np
import pytest

from mlscratch import accuracy_score, cluster_purity, confusion_matrix
from mlscratch.metrics import format_confusion_matrix


def test_accuracy() -> None:
    assert accuracy_score([1, 2, 3, 4], [1, 2, 0, 0]) == 0.5
    assert accuracy_score(["a"], ["a"]) == 1.0


def test_confusion_matrix_hand_checked() -> None:
    y_true = ["cat", "cat", "dog", "dog", "dog", "bird"]
    y_pred = ["cat", "dog", "dog", "dog", "cat", "bird"]
    m, labels = confusion_matrix(y_true, y_pred)
    assert labels.tolist() == ["bird", "cat", "dog"]
    assert m.tolist() == [[1, 0, 0], [0, 1, 1], [0, 1, 2]]
    assert m.sum() == len(y_true)
    assert np.trace(m) / m.sum() == accuracy_score(y_true, y_pred)


def test_confusion_matrix_explicit_labels_order_and_unseen() -> None:
    m, labels = confusion_matrix([1, 2], [1, 3], labels=[3, 2, 1, 0])
    assert labels.tolist() == [3, 2, 1, 0]
    assert m.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]]


def test_confusion_matrix_ignores_unlisted_labels() -> None:
    m, _ = confusion_matrix([1, 2, 9], [1, 2, 9], labels=[1, 2])
    assert m.tolist() == [[1, 0], [0, 1]]


def test_cluster_purity() -> None:
    assert cluster_purity(["a", "a", "b", "b"], [5, 5, 7, 7]) == 1.0
    # Label permutation does not matter.
    assert cluster_purity([0, 0, 1, 1], [1, 1, 0, 0]) == 1.0
    assert cluster_purity(["a", "b", "c"], [0, 0, 0]) == pytest.approx(1 / 3)


@pytest.mark.parametrize("fn", [accuracy_score, confusion_matrix, cluster_purity])
@pytest.mark.parametrize(
    ("a", "b", "match"),
    [
        ([1, 2], [1], "different lengths"),
        ([], [], "empty"),
        ([[1]], [[1]], "1-D"),
    ],
)
def test_label_pair_errors(fn, a, b, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        fn(a, b)


@pytest.mark.parametrize(
    ("labels", "match"),
    [([], "non-empty"), ([[1]], "non-empty"), ([1, 1], "duplicates")],
)
def test_confusion_matrix_bad_labels(labels, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        confusion_matrix([1], [1], labels=labels)


def test_format_confusion_matrix() -> None:
    m, labels = confusion_matrix(["a", "bb", "bb"], ["a", "a", "bb"])
    text = format_confusion_matrix(m, labels)
    lines = text.splitlines()
    assert lines[0].split() == ["true", "\\", "pred", "a", "bb"]
    assert lines[1].split() == ["a", "1", "0"]
    assert lines[2].split() == ["bb", "1", "1"]
    with pytest.raises(ValueError, match="does not match"):
        format_confusion_matrix(m, ["a"])
