"""End-to-end tests for the ``mlscratch`` command-line interface."""

from __future__ import annotations

import io
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mlscratch import make_blobs
from mlscratch.cli import main

ROOT = Path(__file__).resolve().parents[1]
IRIS = ROOT / "data" / "iris.csv"


def run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    code = main(argv, out=out)
    return code, out.getvalue()


def run_subprocess(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "mlscratch", *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )


def parse_accuracy(text: str) -> float:
    line = next(ln for ln in text.splitlines() if ln.startswith("Accuracy:"))
    return float(line.split()[1])


# --------------------------------------------------------------------------- #
# knn on Iris
# --------------------------------------------------------------------------- #
def test_knn_iris_defaults() -> None:
    code, out = run(["knn", str(IRIS)])
    assert code == 0
    assert "150 rows, 4 features, 3 classes" in out
    assert "105 train / 45 test" in out
    assert parse_accuracy(out) >= 0.9
    assert "setosa" in out and "virginica" in out


def test_knn_iris_confusion_matrix_consistent() -> None:
    code, out = run(["knn", str(IRIS), "-k", "7", "--weights", "distance", "--metric", "manhattan"])
    assert code == 0
    lines = out.splitlines()
    start = lines.index("Confusion matrix (rows = true, columns = predicted):") + 2
    matrix = np.array([[int(v) for v in ln.split()[1:]] for ln in lines[start : start + 3]])
    assert matrix.shape == (3, 3)
    assert matrix.sum() == 45
    # Stratified split -> 15 test samples per class.
    assert matrix.sum(axis=1).tolist() == [15, 15, 15]
    assert parse_accuracy(out) == pytest.approx(np.trace(matrix) / 45, abs=1e-4)
    assert "KNNClassifier(k=7, metric='manhattan', weights='distance')" in out


def test_knn_is_deterministic_for_seed() -> None:
    assert run(["knn", str(IRIS), "--seed", "3"]) == run(["knn", str(IRIS), "--seed", "3"])


def test_knn_options_no_scale_no_stratify_label_name() -> None:
    code, out = run(
        ["knn", str(IRIS), "--label", "species", "--no-scale", "--no-stratify", "--test-size", "30"]
    )
    assert code == 0
    assert "120 train / 30 test" in out
    assert "Scaling:   none" in out
    assert "stratified" not in out


def test_knn_temp_file_blobs(tmp_path: Path) -> None:
    X, y = make_blobs(90, 3, cluster_std=0.5, random_state=0)
    p = tmp_path / "blobs.tsv"
    rows = ["\t".join(f"{v:.6f}" for v in x) + f"\tc{c}" for x, c in zip(X, y)]
    p.write_text("\n".join(rows) + "\n")
    code, out = run(["knn", str(p), "--no-header", "--delimiter", "\t", "-k", "3"])
    assert code == 0
    assert parse_accuracy(out) == 1.0


# --------------------------------------------------------------------------- #
# kmeans
# --------------------------------------------------------------------------- #
def test_kmeans_iris() -> None:
    code, out = run(["kmeans", str(IRIS)])
    assert code == 0
    assert "150 rows, 4 features" in out
    inertia = float(next(ln for ln in out.splitlines() if ln.startswith("Inertia:")).split()[1])
    # Known optimum for unscaled Iris with k=3.
    assert inertia == pytest.approx(78.85, abs=0.05)
    sizes_line = next(ln for ln in out.splitlines() if ln.startswith("Cluster sizes:"))
    sizes = sorted(int(part.split(":")[1]) for part in sizes_line.split(":", 1)[1].split(","))
    assert sizes == [38, 50, 62]
    assert "Purity vs 'species': 0.8933" in out
    assert "Iterations (best run):" in out


def test_kmeans_scaled_and_no_label() -> None:
    code, out = run(["kmeans", str(IRIS), "--scale", "-c", "2"])
    assert code == 0 and "standardised" in out
    code, out = run(["kmeans", str(IRIS), "--label", "species", "--no-label", "-c", "2"])
    # --no-label wins, but the label column is non-numeric so loading fails cleanly.
    assert code == 1


def test_kmeans_all_numeric_file_has_no_purity(tmp_path: Path) -> None:
    X, _ = make_blobs(60, 3, random_state=1)
    p = tmp_path / "x.csv"
    np.savetxt(p, X, delimiter=",", header="a,b", comments="")
    code, out = run(["kmeans", str(p), "-c", "3"])
    assert code == 0
    assert "2 features" in out
    assert "Purity" not in out


def test_kmeans_explicit_numeric_label(tmp_path: Path) -> None:
    X, y = make_blobs(60, 3, cluster_std=0.3, random_state=2)
    p = tmp_path / "x.csv"
    np.savetxt(p, np.column_stack([X, y]), delimiter=",", header="a,b,y", comments="")
    code, out = run(["kmeans", str(p), "--label", "y"])
    assert code == 0
    assert "2 features" in out
    assert "Purity vs 'y': 1.0000" in out


# --------------------------------------------------------------------------- #
# Error paths: exit code 1, one-line message, no traceback
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("content", "argv_extra", "match"),
    [
        (None, [], "No such file"),
        ("", [], "empty"),
        ("a,b,y\n1,2,x\n3,4\n", [], "expected 3 fields"),
        ("a,b,y\n1,hello,x\n", [], "expected a number"),
        ("a,b,y\n1,2,x\n", ["--label", "nope"], "not found"),
    ],
)
@pytest.mark.parametrize("command", ["knn", "kmeans"])
def test_bad_input_files(
    tmp_path: Path, capsys, command: str, content: str | None, argv_extra: list[str], match: str
) -> None:
    p = tmp_path / "bad.csv"
    if content is not None:
        p.write_text(content)
    code, out = run([command, str(p), *argv_extra])
    err = capsys.readouterr().err
    assert code == 1
    assert out == ""
    assert err.startswith("mlscratch: error:")
    assert match in err
    assert "Traceback" not in err


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["knn", str(IRIS), "-k", "0"], "k must be >= 1"),
        (["knn", str(IRIS), "-k", "500"], "must be <= the number of training samples"),
        (["knn", str(IRIS), "--test-size", "1.5"], "(0, 1)"),
        (["kmeans", str(IRIS), "-c", "500"], "n_clusters=500"),
        (["kmeans", str(IRIS), "--n-init", "0"], "n_init must be >= 1"),
    ],
)
def test_bad_parameters(capsys, argv: list[str], match: str) -> None:
    code, _ = run(argv)
    err = capsys.readouterr().err
    assert code == 1
    assert match in err


def test_directory_instead_of_file(tmp_path: Path, capsys) -> None:
    code, _ = run(["knn", str(tmp_path)])
    assert code == 1
    assert "mlscratch: error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [[], ["knn"], ["knn", str(IRIS), "--metric", "cosine"], ["bogus"], ["knn", str(IRIS), "--test-size", "x"]],
)
def test_usage_errors_exit_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        main(argv)
    assert info.value.code == 2


# --------------------------------------------------------------------------- #
# Real subprocess: python -m mlscratch
# --------------------------------------------------------------------------- #
def test_subprocess_knn_and_kmeans() -> None:
    knn = run_subprocess("knn", str(IRIS))
    assert knn.returncode == 0, knn.stderr
    assert parse_accuracy(knn.stdout) >= 0.9
    km = run_subprocess("kmeans", str(IRIS))
    assert km.returncode == 0, km.stderr
    assert "Purity" in km.stdout


def test_subprocess_error_has_no_traceback(tmp_path: Path) -> None:
    res = run_subprocess("knn", str(tmp_path / "missing.csv"))
    assert res.returncode == 1
    assert res.stdout == ""
    assert "Traceback" not in res.stderr
    assert res.stderr.strip().startswith("mlscratch: error:")


def test_subprocess_help_and_version() -> None:
    res = run_subprocess("--help")
    assert res.returncode == 0 and "knn" in res.stdout and "kmeans" in res.stdout
    res = run_subprocess("--version")
    assert res.returncode == 0 and "0.1.0" in res.stdout
