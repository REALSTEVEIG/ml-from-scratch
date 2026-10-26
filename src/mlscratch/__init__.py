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
"""

from mlscratch._validation import NotFittedError
from mlscratch.distance import pairwise_distances
from mlscratch.kmeans import KMeans
from mlscratch.knn import KNNClassifier, KNNRegressor

__all__ = [
    "KMeans",
    "KNNClassifier",
    "KNNRegressor",
    "NotFittedError",
    "pairwise_distances",
]

__version__ = "0.1.0"
