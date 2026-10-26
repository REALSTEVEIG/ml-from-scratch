# ml-from-scratch

Two classic machine-learning algorithms, **k-nearest neighbours (kNN)** and
**k-means clustering**, implemented from first principles in Python using
only NumPy.

The goal is to show clear, tested and well-documented numerical code: each
algorithm is written to be easy to read, fully type-hinted, vectorised where
it matters, and strict about its inputs.

## Project layout

```
ml-from-scratch/
├── .github/workflows/ci.yml  lint + tests on Python 3.10-3.13
├── pyproject.toml            packaging, `mlscratch` console script, ruff/pytest config
├── README.md
├── data/
│   └── iris.csv              Fisher's Iris: 150 rows, 4 features, 3 × 50 species
├── src/mlscratch/
│   ├── __init__.py           public API
│   ├── __main__.py           `python -m mlscratch`
│   ├── _validation.py        shared input checks + NotFittedError
│   ├── cli.py                argparse CLI: `knn` and `kmeans` subcommands
│   ├── data.py               load_csv, train_test_split, StandardScaler, make_blobs
│   ├── distance.py           vectorised pairwise distances
│   ├── knn.py                KNNClassifier, KNNRegressor
│   ├── kmeans.py             KMeans (k-means++ init)
│   └── metrics.py            accuracy_score, confusion_matrix, cluster_purity
└── tests/                    pytest suite (unit + end-to-end CLI tests)
```

## Installation

Requires Python 3.10 or newer.

```bash
cd ml-from-scratch
pip install -e ".[dev]"
```

The only runtime dependency is NumPy; `pytest` and `ruff` are installed with
the `dev` extra. Installing also adds an `mlscratch` command.

## Quick start / end-to-end demo

The repository ships with the classic Iris dataset in `data/iris.csv`
(public domain; values follow Fisher's 1936 paper, which corrects two
transcription errors in the common UCI copy).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

ruff check .
pytest
```

**Classify Iris with kNN.** The CLI loads the CSV, makes a stratified,
seeded 70/30 train/test split, standardises features using the training
part only, fits the model and reports test accuracy:

```console
$ mlscratch knn data/iris.csv
Dataset:   data/iris.csv (150 rows, 4 features, 3 classes)
Split:     105 train / 45 test (seed=42, stratified)
Scaling:   standardised (fit on train)
Model:     KNNClassifier(k=5, metric='euclidean', weights='uniform')
Accuracy:  0.9333 (42/45)

Confusion matrix (rows = true, columns = predicted):
true \ pred  setosa  versicolor  virginica
setosa           15           0          0
versicolor        0          14          1
virginica         0           2         13
```

Options: `-k`, `--metric {euclidean,manhattan}`, `--weights
{uniform,distance}`, `--test-size` (fraction or row count), `--seed`,
`--label` (column name or index, default last column), `--no-scale`,
`--no-stratify`, `--no-header`, `--delimiter`.

**Cluster Iris with k-means.** Labels are not used for fitting; if the file
has a label column it is only used to report how pure the clusters are:

```console
$ python -m mlscratch kmeans data/iris.csv
Dataset:   data/iris.csv (150 rows, 4 features)
Scaling:   none
Model:     KMeans(n_clusters=3, max_iter=300, tol=0.0001, n_init=10, random_state=42)
Inertia:   78.8514
Iterations (best run): 6
Cluster sizes: 0: 62, 1: 50, 2: 38
Purity vs 'species': 0.8933

Cluster composition (rows = cluster, columns = label):
cluster  setosa  versicolor  virginica
0             0          48         14
1            50           0          0
2             0           2         36
```

Options: `-c/--clusters`, `--n-init`, `--max-iter`, `--seed`, `--scale`,
`--label`, `--no-label`, `--no-header`, `--delimiter`.

**Bad input fails cleanly** with a one-line message and exit code 1 (usage
mistakes such as an unknown option exit with code 2):

```console
$ mlscratch knn missing.csv
mlscratch: error: missing.csv: No such file or directory
$ mlscratch kmeans bad.csv
mlscratch: error: bad.csv, line 2, column 'b': expected a number, got 'x'.
```

**The same pipeline in Python:**

```python
from mlscratch import (
    KNNClassifier, StandardScaler, accuracy_score, confusion_matrix,
    load_csv, make_blobs, train_test_split,
)

ds = load_csv("data/iris.csv", label_column="species")
X_tr, X_te, y_tr, y_te = train_test_split(ds.X, ds.y, test_size=0.3,
                                          random_state=42, stratify=True)
scaler = StandardScaler().fit(X_tr)
clf = KNNClassifier(k=5).fit(scaler.transform(X_tr), y_tr)
y_pred = clf.predict(scaler.transform(X_te))
accuracy_score(y_te, y_pred)                 # 0.9333...
matrix, labels = confusion_matrix(y_te, y_pred)

X, y = make_blobs(300, centers=3, cluster_std=1.0, random_state=0)  # synthetic data
```

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
- The estimators do no feature scaling themselves. Both algorithms are
  sensitive to features on very different scales, so standardise inputs
  first (`StandardScaler`, or the CLI's default for `knn` / `--scale` for
  `kmeans`).
- k-means uses only Euclidean distance, assumes roughly round,
  similar-sized clusters, and finds local rather than global optima.
- Dense NumPy arrays only; there is no sparse-matrix support.

## Running the tests

```bash
cd ml-from-scratch
ruff check .
python -m pytest
```

The suite covers hand-checkable examples, well-separated blobs (kNN reaches
100% accuracy and k-means recovers the true clusters), reproducibility,
`k=1` memorisation, weighting and tie-breaking, empty-cluster handling,
convergence, and every validation error path. It also checks the bundled
Iris file (150 rows, 50 per species, known values), the data and metric
helpers, and runs the CLI end to end, both in-process and as a real
`python -m mlscratch` subprocess, including bad-file error paths. GitHub
Actions runs ruff and pytest on Python 3.10-3.13 for every push.
