"""mlscratch: classic machine-learning algorithms implemented with NumPy only.

Public API
----------
KNNClassifier, KNNRegressor
    k-nearest-neighbour models for classification and regression.
KMeans
    k-means clustering with k-means++ initialisation.
pairwise_distances
    Vectorised Euclidean / Manhattan distance matrices.
NotFittedError
    Raised when a model is used before ``fit`` has been called.
load_csv, train_test_split, StandardScaler, make_blobs
    Small data utilities (see :mod:`mlscratch.data`).
accuracy_score, confusion_matrix, cluster_purity
    Evaluation metrics (see :mod:`mlscratch.metrics`).
"""

from mlscratch._validation import NotFittedError
from mlscratch.data import (
    DataError,
    Dataset,
    StandardScaler,
    load_csv,
    make_blobs,
    train_test_split,
)
from mlscratch.distance import pairwise_distances
from mlscratch.kmeans import KMeans
from mlscratch.knn import KNNClassifier, KNNRegressor
from mlscratch.metrics import accuracy_score, cluster_purity, confusion_matrix

__all__ = [
    "DataError",
    "Dataset",
    "KMeans",
    "KNNClassifier",
    "KNNRegressor",
    "NotFittedError",
    "StandardScaler",
    "accuracy_score",
    "cluster_purity",
    "confusion_matrix",
    "load_csv",
    "make_blobs",
    "pairwise_distances",
    "train_test_split",
]

__version__ = "0.1.0"
