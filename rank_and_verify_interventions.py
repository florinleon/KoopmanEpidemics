import csv
import json
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from outbreak_ml_utils import DefaultOutbreakThreshold, write_dict_rows_csv
from single_agent_quarantine_search import collect_baseline, make_cfg, metrics_summary, parse_selection, run_counterfactual, write_metrics_csv


DefaultKReport = (1, 5, 10, 20, 50, 100)

ModelDir = "rf_quarantine_tipping_model"
ModelFile = None
CandidateCsv = None
OutputDir = "optimized_quarantine_search"
TopK = 25
NoVerify = False
KReport = ""

# Simulation parameters used when candidates are built from a baseline run, and also for exact verification.
Seed = 50
SusceptibilityMin = 1.3
SusceptibilityMax = 1.3019593
SimDays = 60
Agents = "all"
Days = "0:40"
OutbreakThreshold = DefaultOutbreakThreshold


def load_csv_rows(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        result = float(value)
        if not np.isfinite(result):
            return default
        return result
    except Exception:
        return default


def as_int(value, default=0):
    try:
        return int(round(as_float(value, float(default))))
    except Exception:
        return default


def load_model_and_features(model_dir, model_file=None):
    model_dir = Path(model_dir)
    if model_file is None:
        model_path = model_dir / "quarantine_tipping_random_forest.joblib"
    else:
        model_path = Path(model_file)
    feature_path = model_dir / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}. Run train_quarantine_tipping_rf.py first, or pass --model-file explicitly.")
    if not feature_path.exists():
        raise FileNotFoundError(f"Feature-name file not found: {feature_path}. The ranker needs the exact feature list used during training.")

    model = joblib.load(model_path)
    feature_names = json.loads(feature_path.read_text(encoding="utf-8"))
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError(f"Invalid feature_names.json in {feature_path}")
    return model, [str(x) for x in feature_names]


def build_matrix(rows, feature_names):
    X = np.zeros((len(rows), len(feature_names)), dtype=np.float32)
    missing_counts = {name: 0 for name in feature_names}
    for i, row in enumerate(rows):
        for j, name in enumerate(feature_names):
            if name not in row or row.get(name) in {None, ""}:
                missing_counts[name] += 1
            X[i, j] = float(as_float(row.get(name), 0.0))
    missing_counts = {k: v for k, v in missing_counts.items() if v > 0}
    return X, missing_counts


def positive_score(model, X):
    """Return a sortable score where larger means stronger predicted intervention value."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        if 1 in classes:
            return np.asarray(probs[:, classes.index(1)], dtype=np.float64)
        # A one-class classifier can appear if a very small data set was used.
        return np.zeros(X.shape[0], dtype=np.float64)
    pred = model.predict(X)
    return np.asarray(pred, dtype=np.float64)


def score_rows(rows, model, feature_names):
    X, missing_counts = build_matrix(rows, feature_names)
    scores = positive_score(model, X)
    scored = []
    for idx, row in enumerate(rows):
        out = dict(row)
        out["rf_rank_score"] = float(scores[idx])
        out["rf_rank"] = 0
        scored.append(out)
    scored.sort(key=lambda r: float(r["rf_rank_score"]), reverse=True)
    for rank, row in enumerate(scored, start=1):
        row["rf_rank"] = rank
    return scored, missing_counts


def row_has_exact_outcomes(row):
    return "attack_rate_reduction" in row and "prevented_outbreak" in row


def best_row_by_reduction(rows):
    exact = [r for r in rows if row_has_exact_outcomes(r)]
    if not exact:
        return None
    return max(exact, key=lambda r: as_float(r.get("attack_rate_reduction"), 0.0))


def summarize_ranking(rows, k_values):
    summary = {"n_ranked_rows": int(len(rows)), "has_exact_outcomes": bool(rows and row_has_exact_outcomes(rows[0]))}
    if not summary["has_exact_outcomes"]:
        return summary

    global_best = best_row_by_reduction(rows)
    global_best_reduction = as_float(global_best.get("attack_rate_reduction"), 0.0) if global_best else 0.0
    global_prevented = sum(as_int(r.get("prevented_outbreak"), 0) for r in rows)
    summary.update({"n_prevented_outbreak_total": int(global_prevented), "global_best_attack_rate_reduction": float(global_best_reduction), 
                    "global_best_agent_id": as_int(global_best.get("agent_id"), -1) if global_best else None, 
                    "global_best_day": as_int(global_best.get("day"), -1) if global_best else None, "top_k": []})

    for k in sorted({int(k) for k in k_values if int(k) > 0 and rows}):
        top = list(rows[: min(k, len(rows))])
        best_top = best_row_by_reduction(top)
        best_reduction = as_float(best_top.get("attack_rate_reduction"), 0.0) if best_top else 0.0
        prevented_in_top = sum(as_int(r.get("prevented_outbreak"), 0) for r in top)
        summary["top_k"].append({"k": int(k), "prevented_found": bool(prevented_in_top > 0), "n_prevented_in_top_k": int(prevented_in_top), 
                                 "best_attack_rate_reduction_in_top_k": float(best_reduction), 
                                 "regret_vs_global_best": float(max(0.0, global_best_reduction - best_reduction)), 
                                 "best_agent_id_in_top_k": as_int(best_top.get("agent_id"), -1) if best_top else None, 
                                 "best_day_in_top_k": as_int(best_top.get("day"), -1) if best_top else None})
    return summary


def build_candidate_rows_from_baseline(args, output_dir):
    n_agents = int(make_cfg(args, do_quarantine=False).n_agents)
    agent_ids = parse_selection(args.agents, n_agents)
    days = parse_selection(args.days, int(args.sim_days))

    print("Running one baseline simulation and collecting agent-day snapshots...")
    baseline_metrics, snapshots = collect_baseline(args)
    baseline_summary = metrics_summary(baseline_metrics, n_agents)
    baseline_major = baseline_summary["final_attack_rate"] >= args.outbreak_threshold

    write_metrics_csv(output_dir / "baseline_metrics.csv", baseline_metrics)
    (output_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")

    rows = []
    for agent_id in agent_ids:
        for day in days:
            snapshot = snapshots.get((agent_id, day))
            if snapshot is None:
                continue
            row = dict(snapshot)
            row.update({"seed": int(args.seed), "susceptibility_min": float(args.susceptibility_min), 
                        "susceptibility_max": float(args.susceptibility_max), "sim_days": int(args.sim_days), 
                        "baseline_final_attack_rate": float(baseline_summary["final_attack_rate"]), 
                        "baseline_peak_infected": int(baseline_summary["peak_infected"]), 
                        "baseline_peak_day": int(baseline_summary["peak_day"]), "baseline_is_outbreak": int(bool(baseline_major))})
            rows.append(row)

    meta = {"n_agents_requested": len(agent_ids), "n_days_requested": len(days), "n_candidate_rows": len(rows), 
            "baseline": baseline_summary, "baseline_is_outbreak": int(bool(baseline_major))}
    return rows, meta


def verify_top_k(args, ranked_rows, output_dir):
    n_agents = int(make_cfg(args, do_quarantine=False).n_agents)
    baseline_summary_path = output_dir / "baseline_summary.json"
    if baseline_summary_path.exists():
        baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    else:
        # This fallback is used only if the caller supplies a candidate CSV and asks for verification.
        print("Baseline summary not found in output directory; running baseline once for verification metadata...")
        baseline_metrics, _ = collect_baseline(args)
        baseline_summary = metrics_summary(baseline_metrics, n_agents)
        write_metrics_csv(output_dir / "baseline_metrics.csv", baseline_metrics)
        baseline_summary_path.write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")

    baseline_major = baseline_summary["final_attack_rate"] >= args.outbreak_threshold
    top_rows = list(ranked_rows[: max(0, int(args.top_k))])
    verified = []

    for idx, row in enumerate(top_rows, start=1):
        agent_id = as_int(row.get("agent_id"), -1)
        day = as_int(row.get("day"), -1)
        if agent_id < 0 or day < 0:
            continue
        print(f"[{idx}/{len(top_rows)}] verifying rank={row.get('rf_rank')} agent={agent_id} day={day}")
        cf_metrics = run_counterfactual(args, agent_id, day)
        cf_summary = metrics_summary(cf_metrics, n_agents)
        out = dict(row)
        out.update({"verified_rank_order": int(idx), "intervention_final_attack_rate": float(cf_summary["final_attack_rate"]), "intervention_final_susceptible": int(cf_summary["final_susceptible"]), "intervention_peak_infected": int(cf_summary["peak_infected"]), "intervention_peak_day": int(cf_summary["peak_day"]), "attack_rate_reduction": float(baseline_summary["final_attack_rate"] - cf_summary["final_attack_rate"]), "baseline_is_outbreak": int(bool(baseline_major)), "intervention_is_outbreak": int(cf_summary["final_attack_rate"] >= args.outbreak_threshold), "prevented_outbreak": int(bool(baseline_major and cf_summary["final_attack_rate"] < args.outbreak_threshold))})
        verified.append(out)
        write_dict_rows_csv(output_dir / "verified_top_k.csv", verified)

    return verified


def parse_k_report(spec, top_k):
    values = set(DefaultKReport)
    values.add(int(top_k))
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            values.add(int(part))
    return sorted(k for k in values if k > 0)


def main():
    args = SimpleNamespace(model_dir=ModelDir, model_file=ModelFile, candidate_csv=CandidateCsv, 
                           output_dir=OutputDir, top_k=TopK, no_verify=NoVerify, k_report=KReport, 
                           seed=Seed, susceptibility_min=SusceptibilityMin, susceptibility_max=SusceptibilityMax, 
                           sim_days=SimDays, agents=Agents, days=Days, outbreak_threshold=OutbreakThreshold)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, feature_names = load_model_and_features(args.model_dir, args.model_file)

    if args.candidate_csv:
        print(f"Loading candidates from {args.candidate_csv}...")
        rows = [dict(r) for r in load_csv_rows(args.candidate_csv)]
        candidate_meta = {"candidate_csv": str(Path(args.candidate_csv).resolve()), "n_candidate_rows": len(rows)}
    else:
        rows, candidate_meta = build_candidate_rows_from_baseline(args, output_dir)

    if not rows:
        raise ValueError("No candidate rows were available for ranking.")

    ranked, missing_counts = score_rows(rows, model, feature_names)
    write_dict_rows_csv(output_dir / "ranked_candidates.csv", ranked)

    k_values = parse_k_report(args.k_report, args.top_k)
    retrospective_summary = summarize_ranking(ranked, k_values)
    (output_dir / "ranking_summary.json").write_text(json.dumps(retrospective_summary, indent=2), encoding="utf-8")

    verified = []
    if not args.no_verify and args.top_k > 0:
        verified = verify_top_k(args, ranked, output_dir)
        verified_summary = summarize_ranking(verified, k_values)
        (output_dir / "verified_summary.json").write_text(json.dumps(verified_summary, indent=2), encoding="utf-8")

    summary = {"model_dir": str(Path(args.model_dir).resolve()), 
               "model_file": str(Path(args.model_file).resolve()) if args.model_file else str((Path(args.model_dir) / "quarantine_tipping_random_forest.joblib").resolve()), 
               "n_model_features": int(len(feature_names)), "candidate_meta": candidate_meta, "n_ranked_rows": int(len(ranked)), "top_k_requested": int(args.top_k), 
               "n_verified_rows": int(len(verified)), "missing_model_features_filled_with_zero": missing_counts, 
               "outputs": {"ranked_candidates": str((output_dir / "ranked_candidates.csv").resolve()), 
                           "ranking_summary": str((output_dir / "ranking_summary.json").resolve()), 
                           "verified_top_k": str((output_dir / "verified_top_k.csv").resolve()) if verified else None, 
                           "verified_summary": str((output_dir / "verified_summary.json").resolve()) if verified else None}}
    (output_dir / "rank_and_verify_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    if retrospective_summary.get("has_exact_outcomes"):
        print("\nRetrospective ranking summary:")
        print(json.dumps(retrospective_summary, indent=2))
    if verified:
        print("\nVerified top-K summary:")
        print(json.dumps(summarize_ranking(verified, k_values), indent=2))


if __name__ == "__main__":
    main()
