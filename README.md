# ml-from-scratch

Two classic machine-learning algorithms, **k-nearest neighbours (kNN)** and
**k-means clustering**, implemented from first principles in Python using
only NumPy.

The goal is to show clear, tested and well-documented numerical code: each
algorithm is written to be easy to read, fully type-hinted, vectorised where
it matters, and strict about its inputs.

## Contents

```
ml-from-scratch/
├── pyproject.toml
├── README.md
├── src/mlscratch/
│   ├── __init__.py       public API
│   ├── _validation.py    shared input checks + NotFittedError
│   ├── distance.py       vectorised pairwise distances
│   ├── knn.py            KNNClassifier, KNNRegressor
│   └── kmeans.py         KMeans (k-means++ init)
└── tests/                pytest suite
```

## Installation

Requires Python 3.10 or newer.

```bash
cd ml-from-scratch
pip install -e ".[dev]"
```

The only runtime dependency is NumPy; `pytest` is installed with the `dev`
extra.

## Usage

### k-nearest neighbours

```python
import numpy as np
from mlscratch import KNNClassifier, KNNRegressor

X_train = np.array([[0, 0], [1, 0], [0, 1], [9, 9], [10, 9], [9, 10]])
y_train = np.array(["red", "red", "red", "blue", "blue", "blue"])

clf = KNNClassifier(k=3, metric="euclidean", weights="distance")
clf.fit(X_train, y_train)

clf.predict([[0.5, 0.5], [8, 8]])        # array(['red', 'blue'], ...)
clf.predict_proba([[5, 5]])              # columns follow clf.classes_
clf.score(X_train, y_train)              # accuracy -> 1.0

reg = KNNRegressor(k=2).fit([[0], [1], [10]], [0.0, 2.0, 50.0])
reg.predict([[0.4]])                     # array([1.])
```

### k-means

```python
import numpy as np
from mlscratch import KMeans

rng = np.random.default_rng(0)
X = np.vstack([rng.normal(c, 1.0, size=(50, 2)) for c in ([0, 0], [20, 0], [0, 20])])

km = KMeans(n_clusters=3, n_init=10, random_state=42)
labels = km.fit_predict(X)

km.cluster_centers_   # (3, 2) array of centres
km.inertia_           # sum of squared distances to nearest centre
km.n_iter_            # iterations used by the best run
km.predict([[19, 1]]) # cluster index for new points
```

Calling `predict` before `fit` raises `mlscratch.NotFittedError`.

## How the algorithms work

### k-nearest neighbours

kNN does no real "training". It remembers every labelled example. To
classify a new point, it finds the `k` stored points closest to it and lets
them vote; the most common label wins. With `weights="distance"` closer
neighbours count more (each vote is worth `1 / distance`), so one very close
example can outweigh several distant ones. For regression the prediction is
the (weighted) average of the neighbours' values.

"Closest" depends on the distance metric:

- **Euclidean**: straight-line distance.
- **Manhattan**: the sum of the absolute differences along each axis, like
  walking city blocks.

Ties are resolved deterministically. If two training points are equally
far away, the one that came first in the training data is used. If two
classes get the same vote, the class that sorts first wins. If a query
exactly matches a training point under distance weighting, the exact
matches decide the prediction, which avoids dividing by zero.

### k-means

k-means splits unlabelled data into `k` groups so that each point is close
to the centre of its group. It repeats two steps:

1. **Assign** every point to its nearest centre.
2. **Update** every centre to the mean of the points assigned to it.

It stops when no centre moves more than `tol`, or after `max_iter` rounds.
Each round can only lower the total squared distance (the *inertia*), so
the algorithm always converges, though possibly to a local optimum.

Where the centres start matters a lot. **k-means++** picks the first centre
at random. Each later centre is picked at random as well, but points far
from the existing centres are much more likely to be chosen. This spreads
the starting centres out. The whole procedure is repeated `n_init` times
with different random starts, and the run with the lowest inertia is kept.

If a cluster ends up with no points, its centre is moved to the data point
currently farthest from its own centre. That point is the worst-served in
the dataset, so moving the centre there both fills the cluster and lowers
inertia.

## Complexity

Notation: `n` training samples, `m` query samples, `d` features, `k`
neighbours or clusters, `i` iterations per run.

| Operation | Time | Extra memory |
|---|---|---|
| kNN `fit` | O(1) beyond input validation (O(n·d) to check and store) | O(n·d) stored data |
| kNN `predict` | O(m·n·d) distances + O(m·n log n) neighbour sort | O(m·n) distance matrix |
| k-means++ init | O(n·k·d) | O(n) |
| k-means `fit` | O(n·k·d·i·n_init) | O(n·k) |
| k-means `predict` | O(m·k·d) | O(m·k) |

Notes:

- kNN predict sorts each query row with a stable sort (O(n log n)) to get
  deterministic tie-breaking. Using `np.argpartition` would give O(n) per
  query (or O(n log k) with a heap), but its order among equal distances is
  not guaranteed.
- Euclidean distances use the identity `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b`, so the
  work is one matrix multiplication with O(m·n) memory. Manhattan distance
  uses broadcasting, which needs O(m·n·d) temporary memory.

## Design decisions

- **Scikit-learn-style API** (`fit`/`predict`, trailing-underscore fitted
  attributes) because it is familiar and makes the models easy to compare
  with reference implementations.
- **One validation module.** All checks (2-D numeric input, finite values,
  matching lengths, parameter ranges, feature-count mismatch, fitted state)
  live in `_validation.py`, so every estimator gives the same clear
  `TypeError`/`ValueError` messages.
- **Hyperparameters checked in `__init__`.** A bad `k` or `metric` fails
  immediately. Checks that depend on the data (e.g. `k <= n_samples`) happen
  in `fit`.
- **Reproducibility.** `random_state` accepts `None`, an `int` or a
  `numpy.random.Generator`. With a fixed seed, `KMeans` results are
  bit-for-bit identical across runs.
- **Small shared base class** for the two kNN models instead of a large class
  hierarchy. `KMeans` stands alone because it shares no logic with kNN apart
  from validation and distances.

## Limitations

- Brute-force neighbour search: no KD-tree or ball tree, so kNN prediction
  is slow and memory-hungry for large `n` and `m`. The full `m × n` distance
  matrix is held in memory at once.
- The fast Euclidean formula can lose precision through cancellation when
  feature values are large and points are close together. Distances are
  clamped at zero, so they are never negative.
- No feature scaling is done. Both algorithms are sensitive to features on
  very different scales, so standardise inputs first.
- k-means uses only Euclidean distance, assumes roughly round,
  similar-sized clusters, and finds local rather than global optima.
- Dense NumPy arrays only; there is no sparse-matrix support.

## Running the tests

```bash
cd ml-from-scratch
python -m pytest
```

The suite covers hand-checkable examples, well-separated blobs (kNN reaches
100% accuracy and k-means recovers the true clusters), reproducibility,
`k=1` memorisation, weighting and tie-breaking, empty-cluster handling,
convergence, and every validation error path.
