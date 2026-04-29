import csv
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, average_precision_score, classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import train_test_split
from outbreak_ml_utils import write_dict_rows_csv


Interventions = "quarantine_search/quarantine_counterfactuals.csv"
Target = "prevented_outbreak"
OutputDir = "rf_quarantine_tipping_model"
IncludeIds = False
NEstimators = 600
MaxDepth = None
MinSamplesLeaf = 2
TestSize = 0.20
RandomSeed = 123
Jobs = -1

LeakageColumns = {
    "intervention_final_attack_rate",
    "intervention_final_susceptible",
    "intervention_peak_infected",
    "intervention_peak_day",
    "attack_rate_reduction",
    "intervention_is_outbreak",
    "prevented_outbreak",
}
DefaultExcludedColumns = {
    "agent_id",
    "state",
    "immunity_strength",
}


def load_csv_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value):
    try:
        if value is None or value == "":
            return None
        result = float(value)
        if not np.isfinite(result):
            return None
        return result
    except Exception:
        return None


def choose_feature_names(rows, target, include_ids):
    excluded = set(LeakageColumns)
    excluded.add(target)
    if not include_ids:
        excluded.update(DefaultExcludedColumns)
    candidates = []
    for key in rows[0].keys():
        if key in excluded:
            continue
        values = [as_float(row.get(key)) for row in rows]
        valid = [v for v in values if v is not None]
        if len(valid) == len(rows):
            candidates.append(key)
    return candidates


def build_matrix(rows, feature_names, target):
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    y = np.zeros(len(rows), dtype=np.int64)
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            value = as_float(row.get(name))
            X[i, j] = 0.0 if value is None else float(value)
        target_value = as_float(row.get(target))
        if target_value is None:
            raise ValueError(f"Target column {target!r} has a non-numeric value on row {i}.")
        y[i] = int(round(target_value))
    return X, y


def probability_for_positive(model, X):
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=np.float64)
    return probs[:, classes.index(1)]


def metric_or_none(fn, y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    return float(fn(y_true, y_score))


def main():
    args = SimpleNamespace(interventions=Interventions, target=Target, output_dir=OutputDir, 
                           include_ids=IncludeIds, n_estimators=NEstimators, max_depth=MaxDepth, 
                           min_samples_leaf=MinSamplesLeaf, test_size=TestSize, random_seed=RandomSeed, jobs=Jobs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(args.interventions)
    if not rows:
        raise ValueError("The intervention CSV is empty.")
    if args.target not in rows[0]:
        raise KeyError(f"Target column {args.target!r} not found in {args.interventions}.")

    feature_names = choose_feature_names(rows, args.target, args.include_ids)
    if not feature_names:
        raise ValueError("No numeric feature columns were found after excluding leakage columns.")
    X, y = build_matrix(rows, feature_names, args.target)

    class_values, class_counts = np.unique(y, return_counts=True)
    distribution = {str(int(cls)): int(count) for cls, count in zip(class_values, class_counts)}
    (output_dir / "class_distribution.json").write_text(json.dumps(distribution, indent=2), encoding="utf-8")
    if class_values.size < 2:
        raise ValueError("The target has only one class. Run a broader counterfactual search, adjust the outbreak threshold, or train on multiple baseline parameter regimes.")

    stratify = y if np.min(class_counts) >= 2 else None
    train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size, 
                                           random_state=args.random_seed, stratify=stratify)

    model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, 
                                   min_samples_leaf=args.min_samples_leaf, class_weight="balanced", 
                                   random_state=args.random_seed, n_jobs=args.jobs)
    model.fit(X[train_idx], y[train_idx])

    y_pred = model.predict(X[test_idx])
    y_score = probability_for_positive(model, X[test_idx])
    pr, rc, f1, support = precision_recall_fscore_support(y[test_idx], y_pred, labels=[0, 1], zero_division=0)

    metrics = {
        "interventions": str(Path(args.interventions).resolve()),
        "target": args.target,
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "class_distribution": distribution,
        "accuracy": float(accuracy_score(y[test_idx], y_pred)),
        "roc_auc": metric_or_none(roc_auc_score, y[test_idx], y_score),
        "average_precision": metric_or_none(average_precision_score, y[test_idx], y_score),
        "confusion_matrix": confusion_matrix(y[test_idx], y_pred, labels=[0, 1]).tolist(),
        "class_0_precision": float(pr[0]),
        "class_0_recall": float(rc[0]),
        "class_0_f1": float(f1[0]),
        "class_0_support": int(support[0]),
        "class_1_precision": float(pr[1]),
        "class_1_recall": float(rc[1]),
        "class_1_f1": float(f1[1]),
        "class_1_support": int(support[1]),
        "classification_report": classification_report(y[test_idx], y_pred, zero_division=0),
    }

    joblib.dump(model, output_dir / "quarantine_tipping_random_forest.joblib")
    (output_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    importances = sorted(({"feature": feature_names[idx], "importance": float(value)} for idx, value in enumerate(model.feature_importances_)), key=lambda row: row["importance"], reverse=True)
    write_dict_rows_csv(output_dir / "feature_importance.csv", importances)

    prediction_rows = []
    for local_idx, row_idx in enumerate(test_idx):
        original = dict(rows[int(row_idx)])
        original["split"] = "test"
        original[f"predicted_{args.target}"] = int(y_pred[local_idx])
        original[f"predicted_{args.target}_probability"] = float(y_score[local_idx])
        prediction_rows.append(original)
    write_dict_rows_csv(output_dir / "test_predictions.csv", prediction_rows)

    print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
    print("\nClassification report:\n" + metrics["classification_report"])


if __name__ == "__main__":
    main()
