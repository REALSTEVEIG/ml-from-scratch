"""Command-line interface: ``mlscratch knn ...`` and ``mlscratch kmeans ...``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

import numpy as np

from mlscratch import __version__
from mlscratch.data import DataError, Dataset, StandardScaler, load_csv, train_test_split
from mlscratch.distance import METRICS
from mlscratch.kmeans import KMeans
from mlscratch.knn import WEIGHTS, KNNClassifier
from mlscratch.metrics import (
    accuracy_score,
    cluster_purity,
    confusion_matrix,
    format_confusion_matrix,
)

EXIT_OK = 0
EXIT_DATA_ERROR = 1  # argparse itself uses 2 for usage errors.

_AUTO = "auto"


def _label_arg(value: str) -> str | int:
    """Parse ``--label``: an integer index, a column name, or ``auto``."""
    try:
        return int(value)
    except ValueError:
        return value


def _test_size_arg(value: str) -> float | int:
    """Parse ``--test-size`` as a fraction (contains '.') or a count."""
    try:
        return float(value) if any(ch in value for ch in ".eE") else int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid test size: {value!r}") from None


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argument parser with its subcommands."""
    parser = argparse.ArgumentParser(
        prog="mlscratch",
        description="Run kNN classification or k-means clustering on a CSV file.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="{knn,kmeans}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("csv", help="path to a CSV file of numeric features")
    common.add_argument(
        "--no-header", action="store_true", help="the CSV has no header row"
    )
    common.add_argument(
        "--delimiter", default=",", help="field separator (default: ',')"
    )
    common.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")

    knn = sub.add_parser(
        "knn",
        parents=[common],
        help="train and evaluate a k-nearest-neighbour classifier",
        description="Load CSV -> split -> standardise -> fit kNN -> report test accuracy.",
    )
    knn.add_argument(
        "--label",
        type=_label_arg,
        default=-1,
        help="label column name or index (default: -1, the last column)",
    )
    knn.add_argument("-k", "--k", type=int, default=5, help="number of neighbours (default: 5)")
    knn.add_argument("--metric", choices=METRICS, default="euclidean")
    knn.add_argument("--weights", choices=WEIGHTS, default="uniform")
    knn.add_argument(
        "--test-size",
        type=_test_size_arg,
        default=0.3,
        help="test fraction in (0, 1) or a row count (default: 0.3)",
    )
    knn.add_argument(
        "--no-stratify", action="store_true", help="do not keep class shares equal in the split"
    )
    knn.add_argument(
        "--no-scale", action="store_true", help="skip feature standardisation"
    )
    knn.set_defaults(func=_run_knn)

    km = sub.add_parser(
        "kmeans",
        parents=[common],
        help="cluster the data with k-means",
        description="Load CSV -> fit k-means -> report inertia, iterations, sizes, purity.",
    )
    km.add_argument(
        "--label",
        type=_label_arg,
        default=_AUTO,
        help="label column name or index, used only to score purity "
        "(default: auto = last column if it is non-numeric)",
    )
    km.add_argument(
        "--no-label", action="store_true", help="treat every column as a feature"
    )
    km.add_argument(
        "-c", "--clusters", type=int, default=3, help="number of clusters (default: 3)"
    )
    km.add_argument("--n-init", type=int, default=10, help="number of restarts (default: 10)")
    km.add_argument("--max-iter", type=int, default=300, help="iterations per run (default: 300)")
    km.add_argument(
        "--scale", action="store_true", help="standardise features before clustering"
    )
    km.set_defaults(func=_run_kmeans)
    return parser


def _load(args: argparse.Namespace, label: str | int | None) -> Dataset:
    return load_csv(args.csv, label, header=not args.no_header, delimiter=args.delimiter)


def _run_knn(args: argparse.Namespace, out: TextIO) -> int:
    clf = KNNClassifier(k=args.k, metric=args.metric, weights=args.weights)
    data = _load(args, args.label)
    assert data.y is not None
    X_train, X_test, y_train, y_test = train_test_split(
        data.X,
        data.y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=not args.no_stratify,
    )
    if not args.no_scale:
        scaler = StandardScaler().fit(X_train)  # fit on train only: no test leakage
        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    y_pred = clf.fit(X_train, y_train).predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    matrix, labels = confusion_matrix(y_test, y_pred, labels=np.unique(data.y))

    print(f"Dataset:   {args.csv} ({data.X.shape[0]} rows, {data.X.shape[1]} features, "
          f"{len(labels)} classes)", file=out)
    print(f"Split:     {len(y_train)} train / {len(y_test)} test (seed={args.seed}"
          f"{', stratified' if not args.no_stratify else ''})", file=out)
    print(f"Scaling:   {'none' if args.no_scale else 'standardised (fit on train)'}", file=out)
    print(f"Model:     {clf!r}", file=out)
    print(f"Accuracy:  {acc:.4f} ({int(np.sum(y_pred == y_test))}/{len(y_test)})", file=out)
    print("", file=out)
    print("Confusion matrix (rows = true, columns = predicted):", file=out)
    print(format_confusion_matrix(matrix, labels), file=out)
    return EXIT_OK


def _run_kmeans(args: argparse.Namespace, out: TextIO) -> int:
    km = KMeans(
        n_clusters=args.clusters,
        n_init=args.n_init,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    if args.no_label:
        data = _load(args, None)
    elif args.label == _AUTO:
        data = _load(args, -1)
        assert data.y is not None
        if _all_numeric(data.y):  # last column is a feature, not a label
            data = _load(args, None)
    else:
        data = _load(args, args.label)

    X = StandardScaler().fit_transform(data.X) if args.scale else data.X
    labels = km.fit_predict(X)
    assert km.inertia_ is not None
    sizes = np.bincount(labels, minlength=args.clusters)

    print(f"Dataset:   {args.csv} ({X.shape[0]} rows, {X.shape[1]} features)", file=out)
    print(f"Scaling:   {'standardised' if args.scale else 'none'}", file=out)
    print(f"Model:     {km!r}", file=out)
    print(f"Inertia:   {km.inertia_:.4f}", file=out)
    print(f"Iterations (best run): {km.n_iter_}", file=out)
    print("Cluster sizes: " + ", ".join(f"{i}: {n}" for i, n in enumerate(sizes)), file=out)
    if data.y is not None:
        purity = cluster_purity(data.y, labels)
        print(f"Purity vs '{data.label_name}': {purity:.4f}", file=out)
        print("", file=out)
        print("Cluster composition (rows = cluster, columns = label):", file=out)
        classes = np.unique(data.y)
        table = np.array([[np.sum((labels == c) & (data.y == cls)) for cls in classes]
                          for c in range(args.clusters)])
        print(_format_composition(table, classes), file=out)
    return EXIT_OK


def _all_numeric(values: np.ndarray) -> bool:
    try:
        values.astype(np.float64)
    except ValueError:
        return False
    return True


def _format_composition(table: np.ndarray, classes: np.ndarray) -> str:
    names = [str(c) for c in classes.tolist()]
    head = "cluster"
    widths = [max(len(n), len(str(int(table[:, j].max())))) for j, n in enumerate(names)]
    lines = ["  ".join([head, *(n.rjust(w) for n, w in zip(names, widths))])]
    for i, row in enumerate(table):
        cells = (str(int(v)).rjust(w) for v, w in zip(row, widths))
        lines.append("  ".join([str(i).ljust(len(head)), *cells]))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None, out: TextIO | None = None) -> int:
    """Run the CLI and return a process exit code.

    Input problems (missing or malformed files, invalid parameters) are
    reported as a one-line ``mlscratch: error: ...`` on stderr with exit
    code 1 rather than a traceback.
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    stream = out if out is not None else sys.stdout
    try:
        return int(args.func(args, stream))
    except (OSError, DataError, ValueError, TypeError) as exc:
        message = exc.strerror if isinstance(exc, OSError) and exc.strerror else str(exc)
        if isinstance(exc, OSError) and exc.filename is not None:
            message = f"{exc.filename}: {message}"
        print(f"mlscratch: error: {message}", file=sys.stderr)
        return EXIT_DATA_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
