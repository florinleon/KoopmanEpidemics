import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from outbreak_ml_utils import DefaultOutbreakThreshold, build_outbreak_window_table, load_npz_dataset, write_dict_rows_csv


Dataset = "baseline_dataset.npz"
KoopmanDir = "koopman_training"
RequireKoopman = False
OutputDir = "rf_outbreak_model"
HistoryDays = 5
EndDayMin = None
EndDayMax = 12
OutbreakThreshold = DefaultOutbreakThreshold
NEstimators = 600
MaxDepth = None
MinSamplesLeaf = 2
TestSize = 0.20
RandomSeed = 123
Jobs = -1


def _probability_for_positive(model, X):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=np.float64)
    return probs[:, classes.index(1)]


def _safe_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_average_precision(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _split_indices(y, groups, test_size, random_seed):
    unique_groups = np.unique(groups)
    if unique_groups.size >= 3:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_idx, test_idx = next(splitter.split(np.zeros_like(y), y, groups=groups))
        return train_idx, test_idx
    return train_test_split(np.arange(len(y)), test_size=test_size, random_state=random_seed, stratify=y if len(np.unique(y)) > 1 else None)


def train_and_evaluate(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_npz_dataset(args.dataset)
    koopman_dir = Path(args.koopman_dir) if args.koopman_dir else None
    if koopman_dir is not None and not koopman_dir.exists():
        if args.require_koopman:
            raise FileNotFoundError(f"Koopman directory not found: {koopman_dir}")
        koopman_dir = None

    X, y, records, feature_names = \
        build_outbreak_window_table(data, history_days=args.history_days, end_day_min=args.end_day_min, 
                                    end_day_max=args.end_day_max, outbreak_threshold=args.outbreak_threshold, 
                                    koopman_training_dir=koopman_dir, require_koopman=args.require_koopman)

    groups = np.asarray([r["run_index"] for r in records], dtype=np.int64)
    if len(np.unique(y)) < 2:
        raise ValueError("Only one class is present in the feature table. Adjust the outbreak threshold or add runs from both contained and outbreak regimes.")

    train_idx, test_idx = _split_indices(y, groups, args.test_size, args.random_seed)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, 
                                   min_samples_leaf=args.min_samples_leaf, class_weight="balanced", 
                                   random_state=args.random_seed, n_jobs=args.jobs)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = _probability_for_positive(model, X_test)
    pr, rc, f1, support = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)

    metrics = {
        "dataset": str(Path(args.dataset).resolve()),
        "koopman_dir": None if koopman_dir is None else str(koopman_dir.resolve()),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "n_train_rows": int(X_train.shape[0]),
        "n_test_rows": int(X_test.shape[0]),
        "n_runs": int(len(np.unique(groups))),
        "outbreak_threshold": float(args.outbreak_threshold),
        "history_days": int(args.history_days),
        "end_day_min": None if args.end_day_min is None else int(args.end_day_min),
        "end_day_max": int(args.end_day_max),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": _safe_auc(y_test, y_score),
        "average_precision": _safe_average_precision(y_test, y_score),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=[0, 1]).tolist(),
        "class_0_precision": float(pr[0]),
        "class_0_recall": float(rc[0]),
        "class_0_f1": float(f1[0]),
        "class_0_support": int(support[0]),
        "class_1_precision": float(pr[1]),
        "class_1_recall": float(rc[1]),
        "class_1_f1": float(f1[1]),
        "class_1_support": int(support[1]),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    joblib.dump(model, output_dir / "outbreak_random_forest.joblib")
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importances = sorted(({"feature": feature_names[idx], "importance": float(value)} for idx, value in enumerate(model.feature_importances_)), key=lambda row: row["importance"], reverse=True)
    write_dict_rows_csv(output_dir / "feature_importance.csv", importances)

    prediction_rows = []
    for local_idx, row_idx in enumerate(test_idx):
        row = dict(records[int(row_idx)])
        row["split"] = "test"
        row["predicted_outbreak_label"] = int(y_pred[local_idx])
        row["predicted_outbreak_probability"] = float(y_score[local_idx])
        prediction_rows.append(row)
    write_dict_rows_csv(output_dir / "test_predictions.csv", prediction_rows)
    return metrics


def main():
    args = SimpleNamespace(dataset=Dataset, koopman_dir=KoopmanDir, require_koopman=RequireKoopman, 
                           output_dir=OutputDir, history_days=HistoryDays, end_day_min=EndDayMin, 
                           end_day_max=EndDayMax, outbreak_threshold=OutbreakThreshold, 
                           n_estimators=NEstimators, max_depth=MaxDepth, min_samples_leaf=MinSamplesLeaf, 
                           test_size=TestSize, random_seed=RandomSeed, jobs=Jobs)
    if args.koopman_dir == "":
        args.koopman_dir = None
    metrics = train_and_evaluate(args)
    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("\nClassification report:\n" + metrics["classification_report"])


if __name__ == "__main__":
    main()
