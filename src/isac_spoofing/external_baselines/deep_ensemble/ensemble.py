"""Multi-BS ensemble candidates from the source Deep Ensemble method."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DEFAULT_CANDIDATES = (
    "mlp",
    "decision_tree",
    "logistic_regression",
    "svm_linear",
    "svm_rbf",
    "gaussian_nb",
)


def _sklearn_components():
    try:
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as error:
        raise RuntimeError(
            "Multi-BS ensemble training requires scikit-learn and scipy"
        ) from error
    return {
        "joblib": joblib,
        "LogisticRegression": LogisticRegression,
        "GaussianNB": GaussianNB,
        "MLPClassifier": MLPClassifier,
        "make_pipeline": make_pipeline,
        "StandardScaler": StandardScaler,
        "SVC": SVC,
        "DecisionTreeClassifier": DecisionTreeClassifier,
    }


def build_ensemble_candidate(name: str, seed: int, input_dim: int):
    if input_dim < 1:
        raise ValueError("input_dim must be positive")
    components = _sklearn_components()
    make_pipeline = components["make_pipeline"]
    standardizer = components["StandardScaler"]
    if name == "logistic_regression":
        return make_pipeline(
            standardizer(),
            components["LogisticRegression"](
                C=0.01,
                max_iter=500,
                random_state=seed,
            ),
        )
    if name == "decision_tree":
        return components["DecisionTreeClassifier"](
            max_depth=input_dim,
            random_state=seed,
        )
    if name == "svm_linear":
        return make_pipeline(
            standardizer(),
            components["SVC"](
                kernel="linear",
                C=0.025,
                probability=True,
                random_state=seed,
            ),
        )
    if name == "svm_rbf":
        return make_pipeline(
            standardizer(),
            components["SVC"](
                kernel="rbf",
                C=1.0,
                gamma=2.0,
                probability=True,
                random_state=seed,
            ),
        )
    if name == "gaussian_nb":
        return components["GaussianNB"]()
    if name == "mlp":
        return make_pipeline(
            standardizer(),
            components["MLPClassifier"](
                hidden_layer_sizes=(input_dim,),
                activation="relu",
                solver="adam",
                learning_rate_init=0.005,
                batch_size=1024,
                max_iter=500,
                early_stopping=True,
                n_iter_no_change=100,
                random_state=seed,
            ),
        )
    raise ValueError(f"unknown ensemble candidate {name!r}")


def train_ensemble_candidates(
    bs_probabilities: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    candidate_names: tuple[str, ...] = DEFAULT_CANDIDATES,
    seed: int = 20260714,
    max_svm_samples: int = 20000,
    resume: bool = True,
) -> dict[str, object]:
    """Fit candidate meta-classifiers and save each candidate independently."""

    values = np.asarray(bs_probabilities, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 2 or targets.shape != (values.shape[0],):
        raise ValueError("ensemble inputs require shape (samples,BS) and labels")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    components = _sklearn_components()
    joblib = components["joblib"]
    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for candidate_index, name in enumerate(candidate_names):
        path = output_dir / f"{name}.joblib"
        if resume and path.exists():
            records.append(
                {
                    "name": name,
                    "path": path.name,
                    "training_samples": None,
                    "status": "reused",
                }
            )
            continue
        model = build_ensemble_candidate(
            name,
            seed + candidate_index,
            input_dim=values.shape[1],
        )
        fit_values = values
        fit_targets = targets
        if name.startswith("svm_") and values.shape[0] > max_svm_samples:
            indices = rng.choice(
                values.shape[0],
                size=max_svm_samples,
                replace=False,
            )
            fit_values = values[indices]
            fit_targets = targets[indices]
        print(
            f"[Ensemble][{name}] training on {fit_targets.size} samples "
            f"with {values.shape[1]} BS inputs",
            flush=True,
        )
        model.fit(fit_values, fit_targets)
        joblib.dump(model, path)
        if name == "mlp":
            classifier = model.named_steps["mlpclassifier"]
            print(
                f"[Ensemble][{name}] completed after "
                f"{classifier.n_iter_} iterations, loss={classifier.loss_:.6f}",
                flush=True,
            )
        records.append(
            {
                "name": name,
                "path": path.name,
                "training_samples": int(fit_targets.size),
                "status": "trained",
            }
        )
    payload = {
        "candidate_names": list(candidate_names),
        "seed": int(seed),
        "max_svm_samples": int(max_svm_samples),
        "input_dim": int(values.shape[1]),
        "paper_mlp_architecture": [
            int(values.shape[1]),
            int(values.shape[1]),
            1,
        ],
        "candidates": records,
    }
    (output_dir / "candidates.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return payload


def load_ensemble(path: Path):
    return _sklearn_components()["joblib"].load(path)


def predict_positive_probability(model, values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(values), dtype=np.float64)
    classes = np.asarray(model.classes_)
    positive_indices = np.flatnonzero(classes == 1)
    if positive_indices.size != 1:
        raise ValueError("ensemble model does not expose a unique positive class")
    return probabilities[:, int(positive_indices[0])]
