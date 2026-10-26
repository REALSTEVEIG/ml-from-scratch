"""Evaluation metrics: accuracy, confusion matrix and cluster purity."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _check_label_pair(y_true: ArrayLike, y_pred: ArrayLike) -> tuple[NDArray[Any], NDArray[Any]]:
    """Validate two 1-D, non-empty label vectors of equal length."""
    t = np.asarray(y_true)
    p = np.asarray(y_pred)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError(
            f"y_true and y_pred must be 1-D, got {t.ndim} and {p.ndim} dimension(s)."
        )
    if t.shape[0] != p.shape[0]:
        raise ValueError(
            f"y_true and y_pred have different lengths: {t.shape[0]} and {p.shape[0]}."
        )
    if t.shape[0] == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return t, p


def accuracy_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraction of predictions that equal the true labels.

    >>> accuracy_score([0, 1, 1, 0], [0, 1, 0, 0])
    0.75
    """
    t, p = _check_label_pair(y_true, y_pred)
    return float(np.mean(t == p))


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, labels: ArrayLike | None = None
) -> tuple[NDArray[np.intp], NDArray[Any]]:
    """Count true-vs-predicted label pairs.

    Parameters
    ----------
    y_true, y_pred:
        1-D label vectors of equal length.
    labels:
        Order of the rows / columns. Defaults to the sorted union of labels
        in ``y_true`` and ``y_pred``. Pairs whose labels are not listed are
        ignored.

    Returns
    -------
    matrix, labels
        ``matrix[i, j]`` is the number of samples with true label
        ``labels[i]`` predicted as ``labels[j]``.

    >>> m, lab = confusion_matrix(["a", "b", "b"], ["a", "a", "b"])
    >>> m.tolist(), lab.tolist()
    ([[1, 0], [1, 1]], ['a', 'b'])
    """
    t, p = _check_label_pair(y_true, y_pred)
    if labels is None:
        label_arr = np.unique(np.concatenate([t, p]))
    else:
        label_arr = np.asarray(labels)
        if label_arr.ndim != 1 or label_arr.size == 0:
            raise ValueError("labels must be a non-empty 1-D array.")
        if np.unique(label_arr).size != label_arr.size:
            raise ValueError("labels must not contain duplicates.")
    index = {label: i for i, label in enumerate(label_arr.tolist())}
    matrix = np.zeros((label_arr.size, label_arr.size), dtype=np.intp)
    for a, b in zip(t.tolist(), p.tolist()):
        if a in index and b in index:
            matrix[index[a], index[b]] += 1
    return matrix, label_arr


def cluster_purity(y_true: ArrayLike, clusters: ArrayLike) -> float:
    """Share of samples whose cluster's majority true label matches their own.

    1.0 means every cluster contains a single class. Purity never decreases
    as the number of clusters grows, so compare it only at a fixed ``k``.

    >>> cluster_purity(["a", "a", "b", "b"], [0, 0, 0, 1])
    0.75
    """
    t, c = _check_label_pair(y_true, clusters)
    matrix, _ = _contingency(c, t)
    return float(matrix.max(axis=1).sum() / t.shape[0])


def _contingency(rows: NDArray[Any], cols: NDArray[Any]) -> tuple[NDArray[np.intp], NDArray[Any]]:
    """Contingency table with rows = unique ``rows`` and cols = unique ``cols``."""
    r_labels, r_idx = np.unique(rows, return_inverse=True)
    c_labels, c_idx = np.unique(cols, return_inverse=True)
    table = np.zeros((r_labels.size, c_labels.size), dtype=np.intp)
    np.add.at(table, (r_idx, c_idx), 1)
    return table, r_labels


def format_confusion_matrix(matrix: NDArray[np.intp], labels: ArrayLike) -> str:
    """Render a confusion matrix as an aligned text table (rows = true)."""
    names = [str(v) for v in np.asarray(labels).tolist()]
    if matrix.shape != (len(names), len(names)):
        raise ValueError(
            f"matrix shape {matrix.shape} does not match {len(names)} labels."
        )
    head = "true \\ pred"
    width0 = max(len(head), *(len(n) for n in names))
    widths = [max(len(n), len(str(int(matrix[:, j].max())))) for j, n in enumerate(names)]
    lines = [
        "  ".join([head.ljust(width0), *(n.rjust(w) for n, w in zip(names, widths))])
    ]
    for i, name in enumerate(names):
        cells = (str(int(matrix[i, j])).rjust(w) for j, w in enumerate(widths))
        lines.append("  ".join([name.ljust(width0), *cells]))
    return "\n".join(lines)
