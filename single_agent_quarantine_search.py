import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import joblib
import numpy as np
from scheduler import run_simulation
from outbreak_ml_utils import DefaultOutbreakThreshold, agent_snapshot_features, make_config_clone, write_dict_rows_csv


Seed = 50
SusceptibilityMin = 1.3
SusceptibilityMax = 1.3019593
SimDays = 60
Agents = "all"
Days = "0:40"
OutbreakThreshold = DefaultOutbreakThreshold
OutputDir = "quarantine_search"
StopAfterFirstTipping = False
OutbreakRfModelDir = "rf_outbreak_model"
OutbreakRfModelFile = "outbreak_random_forest.joblib"
OutbreakRfProbabilityThreshold = 0.5
OutbreakRfHistoryDays = None
OutbreakRfEndDay = None
SkipOutbreakRfGate = False
ForceSearch = False


def parse_selection(spec, upper):
    """Parse selections such as 'all', '0,4,9', or '0:40'.

    Colon ranges are half-open, matching Python range semantics. Thus 0:40
    means days or agents 0 through 39.
    """
    spec = str(spec).strip().lower()
    if spec in {"all", "*"}:
        return list(range(upper))
    result = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            bits = part.split(":")
            if len(bits) not in {2, 3}:
                raise ValueError(f"Invalid range selection: {part}")
            start = int(bits[0]) if bits[0] else 0
            stop = int(bits[1]) if bits[1] else upper
            step = int(bits[2]) if len(bits) == 3 and bits[2] else 1
            result.extend(range(start, stop, step))
        else:
            result.append(int(part))
    result = sorted({x for x in result if 0 <= x < upper})
    if not result:
        raise ValueError(f"Selection produced no valid indices: {spec}")
    return result


def make_cfg(args, *, do_quarantine):
    return make_config_clone(seed=args.seed, susceptibility_min=args.susceptibility_min, 
                             susceptibility_max=args.susceptibility_max, sim_days=args.sim_days, do_quarantine=do_quarantine)


def metrics_summary(metrics, n_agents):
    if not metrics.get("susceptible"):
        return {
            "n_days_recorded": 0,
            "final_susceptible": int(n_agents),
            "final_attack_rate": 0.0,
            "peak_infected": 0,
            "peak_day": -1,
        }
    infected = list(metrics["infected"])
    peak_infected = max(infected) if infected else 0
    peak_day = infected.index(peak_infected) if infected else -1
    final_susceptible = int(metrics["susceptible"][-1])
    return {
        "n_days_recorded": int(len(metrics["susceptible"])),
        "final_susceptible": final_susceptible,
        "final_attack_rate": float((n_agents - final_susceptible) / n_agents),
        "peak_infected": int(peak_infected),
        "peak_day": int(peak_day),
    }


def write_metrics_csv(path, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(metrics.keys())
    max_len = max((len(metrics[k]) for k in keys), default=0)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["day"] + keys)
        for idx in range(max_len):
            writer.writerow([idx] + [metrics[k][idx] if idx < len(metrics[k]) else "" for k in keys])


def collect_baseline(args):
    cfg = make_cfg(args, do_quarantine=False)
    snapshots = {}

    def day_start_recorder(day, agents):
        for agent in agents:
            snapshots[(int(agent.agent_id), int(day))] = agent_snapshot_features(agent, day)

    metrics = run_simulation(cfg, day_start_recorder=day_start_recorder)
    return metrics, snapshots


def run_counterfactual(args, agent_id, day):
    cfg = make_cfg(args, do_quarantine=True)

    def quarantine_fn(current_day, current_agent_id):
        return current_day == day and current_agent_id == agent_id

    return run_simulation(cfg, quarantine_fn=quarantine_fn)


def _safe_slope(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    x = np.arange(values.size, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.dot(x, x))
    if denom <= 0:
        return 0.0
    y = values - values.mean()
    return float(np.dot(x, y) / denom)


def _add_series_features(features, name, values, n_agents):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return
    last = float(values[-1])
    first = float(values[0])
    features[f"{name}__last"] = last
    features[f"{name}__first"] = first
    features[f"{name}__delta"] = last - first
    features[f"{name}__mean"] = float(np.mean(values))
    features[f"{name}__max"] = float(np.max(values))
    features[f"{name}__min"] = float(np.min(values))
    features[f"{name}__std"] = float(np.std(values))
    features[f"{name}__sum"] = float(np.sum(values))
    features[f"{name}__slope"] = _safe_slope(values)
    if n_agents > 0:
        inv = 1.0 / float(n_agents)
        features[f"{name}__last_frac"] = last * inv
        features[f"{name}__delta_frac"] = (last - first) * inv
        features[f"{name}__mean_frac"] = float(np.mean(values)) * inv
        features[f"{name}__max_frac"] = float(np.max(values)) * inv
        features[f"{name}__sum_per_agent"] = float(np.sum(values)) * inv


def _ratio(num, den):
    return float(num) / float(den + 1e-9)


def _positive_class_probability(model, X):
    if not hasattr(model, "predict_proba"):
        return float(model.predict(X)[0])
    probs = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        return 0.0
    return float(probs[0, classes.index(1)])


def _infer_outbreak_rf_settings(model_dir, args):
    """Return the history length and end day used to build the outbreak RF gate features."""
    history_days = args.outbreak_rf_history_days
    end_day = args.outbreak_rf_end_day
    metrics_path = model_dir / "metrics.json"

    if metrics_path.exists():
        try:
            training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if history_days is None and training_metrics.get("history_days") is not None:
                history_days = int(training_metrics["history_days"])
            if end_day is None and training_metrics.get("end_day_max") is not None:
                end_day = int(training_metrics["end_day_max"])
        except Exception as exc:
            print(f"Warning: could not read outbreak RF settings from {metrics_path}: {exc}")

    if history_days is None:
        history_days = 5
    if end_day is None:
        end_day = history_days - 1
    if history_days <= 0:
        raise ValueError("--outbreak-rf-history-days must be positive.")
    if end_day < history_days - 1:
        raise ValueError("--outbreak-rf-end-day must be at least history_days - 1.")
    return int(history_days), int(end_day)


def build_outbreak_rf_feature_row(metrics, *, n_agents, seed, susceptibility_min, susceptibility_max, history_days, end_day):
    """Build one feature row compatible with train_outbreak_rf.py.

    The outbreak RF was trained on window-level aggregate observables. This
    function reproduces that feature construction from a baseline simulation's
    recorded daily metrics.
    """
    if not metrics:
        raise ValueError("No baseline metrics are available for outbreak RF prediction.")

    max_len = max((len(v) for v in metrics.values()), default=0)
    if max_len == 0:
        raise ValueError("No baseline metric rows are available for outbreak RF prediction.")
    if end_day >= max_len:
        raise ValueError(
            f"Outbreak RF prediction end day {end_day} is unavailable; baseline only recorded days 0 through {max_len - 1}."
        )

    start = end_day - history_days + 1
    features = {
        "run_index_numeric": 0.0,
        "seed_numeric": float(seed),
        "end_day": float(end_day),
        "history_days": float(history_days),
        "susceptibility_min": float(susceptibility_min),
        "susceptibility_max": float(susceptibility_max),
    }

    for name, series in metrics.items():
        values = np.asarray(series[start:end_day + 1], dtype=np.float64)
        _add_series_features(features, str(name), values, int(n_agents))

    if "infected" in metrics and "susceptible" in metrics:
        infected_last = float(np.asarray(metrics["infected"], dtype=np.float64)[end_day])
        susceptible_last = float(np.asarray(metrics["susceptible"], dtype=np.float64)[end_day])
        features["infected_to_susceptible_last"] = _ratio(infected_last, susceptible_last)
    if "new_infections" in metrics and "infected" in metrics:
        new_last = float(np.asarray(metrics["new_infections"], dtype=np.float64)[end_day])
        infected_last = float(np.asarray(metrics["infected"], dtype=np.float64)[end_day])
        features["new_infections_to_infected_last"] = _ratio(new_last, infected_last)
    if "infected_mobile" in metrics and "infected" in metrics:
        mobile_last = float(np.asarray(metrics["infected_mobile"], dtype=np.float64)[end_day])
        infected_last = float(np.asarray(metrics["infected"], dtype=np.float64)[end_day])
        features["infected_mobile_share_last"] = _ratio(mobile_last, infected_last)

    # The exhaustive-search script does not reconstruct Koopman latent features
    # for this fresh baseline trajectory. If the outbreak RF was trained with
    # Koopman columns, they will be filled as 0.0 below and koopman_available=0.
    features["koopman_available"] = 0.0
    return features


def predict_baseline_outbreak_with_rf(args, baseline_metrics, n_agents):
    model_dir = Path(args.outbreak_rf_model_dir)
    model_path = model_dir / args.outbreak_rf_model_file
    feature_path = model_dir / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Outbreak RF model not found: {model_path}")
    if not feature_path.exists():
        raise FileNotFoundError(f"Outbreak RF feature list not found: {feature_path}")

    model = joblib.load(model_path)
    feature_names = [str(x) for x in json.loads(feature_path.read_text(encoding="utf-8"))]
    history_days, end_day = _infer_outbreak_rf_settings(model_dir, args)

    row = build_outbreak_rf_feature_row(baseline_metrics, n_agents=n_agents, seed=args.seed, susceptibility_min=args.susceptibility_min, susceptibility_max=args.susceptibility_max, history_days=history_days, end_day=end_day)

    X = np.zeros((1, len(feature_names)), dtype=np.float32)
    missing_features = []
    for col_idx, name in enumerate(feature_names):
        value = row.get(name, None)
        if value is None or not math.isfinite(float(value)):
            missing_features.append(name)
            value = 0.0
        X[0, col_idx] = float(value)

    probability = _positive_class_probability(model, X)
    predicted_by_threshold = int(probability >= float(args.outbreak_rf_probability_threshold))
    try:
        raw_predicted_label = int(model.predict(X)[0])
    except Exception:
        raw_predicted_label = predicted_by_threshold

    return {
        "enabled": True,
        "model_dir": str(model_dir),
        "model_file": str(model_path),
        "feature_names_file": str(feature_path),
        "history_days": int(history_days),
        "end_day": int(end_day),
        "probability_threshold": float(args.outbreak_rf_probability_threshold),
        "predicted_outbreak_probability": float(probability),
        "predicted_outbreak": int(predicted_by_threshold),
        "raw_model_predicted_label": int(raw_predicted_label),
        "n_model_features": int(len(feature_names)),
        "n_missing_features": int(len(missing_features)),
        "n_missing_koopman_features": int(sum(name.startswith("koopman_") for name in missing_features)),
        "missing_features_preview": missing_features[:25],
        "warning": (
            "Some model features were unavailable and were filled with zero. "
            "This is expected for Koopman latent features unless they are computed for this baseline."
            if missing_features
            else None
        ),
    }


def _write_empty_counterfactual_csv(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["message"])
        writer.writerow(["Search skipped because the outbreak RF did not predict an outbreak."])


def main():
    args = SimpleNamespace(seed=Seed, susceptibility_min=SusceptibilityMin, susceptibility_max=SusceptibilityMax, 
                           sim_days=SimDays, agents=Agents, days=Days, outbreak_threshold=OutbreakThreshold, 
                           output_dir=OutputDir, stop_after_first_tipping=StopAfterFirstTipping, 
                           outbreak_rf_model_dir=OutbreakRfModelDir, outbreak_rf_model_file=OutbreakRfModelFile, 
                           outbreak_rf_probability_threshold=OutbreakRfProbabilityThreshold, outbreak_rf_history_days=OutbreakRfHistoryDays, 
                           outbreak_rf_end_day=OutbreakRfEndDay, skip_outbreak_rf_gate=SkipOutbreakRfGate, force_search=ForceSearch)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_agents = int(make_cfg(args, do_quarantine=False).n_agents)
    agent_ids = parse_selection(args.agents, n_agents)
    days = parse_selection(args.days, int(args.sim_days))

    print("Running baseline simulation and collecting agent-day snapshots...")
    baseline_metrics, snapshots = collect_baseline(args)
    baseline_summary = metrics_summary(baseline_metrics, n_agents)
    baseline_major = baseline_summary["final_attack_rate"] >= args.outbreak_threshold

    write_metrics_csv(output_dir / "baseline_metrics.csv", baseline_metrics)
    (output_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")

    rf_gate_summary = {"enabled": False}
    if not args.skip_outbreak_rf_gate:
        print("Scoring baseline trajectory with the outbreak random forest...")
        rf_gate_summary = predict_baseline_outbreak_with_rf(args, baseline_metrics, n_agents)
        (output_dir / "outbreak_rf_gate_summary.json").write_text(json.dumps(rf_gate_summary, indent=2), encoding="utf-8")
        print(json.dumps(rf_gate_summary, indent=2))

        if not bool(rf_gate_summary["predicted_outbreak"]) and not args.force_search:
            csv_path = output_dir / "quarantine_counterfactuals.csv"
            _write_empty_counterfactual_csv(csv_path)
            summary = {
                "n_agents_requested": len(agent_ids),
                "n_days_requested": len(days),
                "n_counterfactuals_completed": 0,
                "search_skipped": True,
                "skip_reason": "outbreak_rf_predicted_no_outbreak",
                "baseline": baseline_summary,
                "outbreak_threshold": float(args.outbreak_threshold),
                "outbreak_rf_gate": rf_gate_summary,
                "n_prevented_outbreak": 0,
                "best_attack_rate_reduction": 0.0,
            }
            (output_dir / "quarantine_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print("Outbreak RF predicted no outbreak; skipped agent-by-agent quarantine search.")
            print(json.dumps(summary, indent=2))
            print(f"Wrote skip marker to: {csv_path}")
            return

    if not baseline_major and not args.force_search:
        csv_path = output_dir / "quarantine_counterfactuals.csv"
        _write_empty_counterfactual_csv(csv_path)
        summary = {
            "n_agents_requested": len(agent_ids),
            "n_days_requested": len(days),
            "n_counterfactuals_completed": 0,
            "search_skipped": True,
            "skip_reason": "full_baseline_not_outbreak",
            "baseline": baseline_summary,
            "outbreak_threshold": float(args.outbreak_threshold),
            "outbreak_rf_gate": rf_gate_summary,
            "n_prevented_outbreak": 0,
            "best_attack_rate_reduction": 0.0,
        }
        (output_dir / "quarantine_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print("Full baseline was not an outbreak; skipped agent-by-agent quarantine search.")
        print(json.dumps(summary, indent=2))
        print(f"Wrote skip marker to: {csv_path}")
        return

    rows = []
    csv_path = output_dir / "quarantine_counterfactuals.csv"

    total = len(agent_ids) * len(days)
    completed = 0
    found_tipping = False

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = None
        for agent_id in agent_ids:
            for day in days:
                completed += 1
                snapshot = snapshots.get((agent_id, day))
                if snapshot is None:
                    # The baseline may have stopped before this day. Such late
                    # interventions cannot affect the already-contained run.
                    continue

                print(f"[{completed}/{total}] quarantine agent={agent_id} day={day}")
                cf_metrics = run_counterfactual(args, agent_id, day)
                cf_summary = metrics_summary(cf_metrics, n_agents)

                row = dict(snapshot)
                row.update(
                    {
                        "seed": int(args.seed),
                        "susceptibility_min": float(args.susceptibility_min),
                        "susceptibility_max": float(args.susceptibility_max),
                        "sim_days": int(args.sim_days),
                        "outbreak_rf_predicted_outbreak": int(rf_gate_summary.get("predicted_outbreak", -1)),
                        "outbreak_rf_predicted_probability": float(rf_gate_summary.get("predicted_outbreak_probability", math.nan)),
                        "baseline_final_attack_rate": float(baseline_summary["final_attack_rate"]),
                        "baseline_peak_infected": int(baseline_summary["peak_infected"]),
                        "baseline_peak_day": int(baseline_summary["peak_day"]),
                        "intervention_final_attack_rate": float(cf_summary["final_attack_rate"]),
                        "intervention_final_susceptible": int(cf_summary["final_susceptible"]),
                        "intervention_peak_infected": int(cf_summary["peak_infected"]),
                        "intervention_peak_day": int(cf_summary["peak_day"]),
                        "attack_rate_reduction": float(baseline_summary["final_attack_rate"] - cf_summary["final_attack_rate"]),
                        "baseline_is_outbreak": int(bool(baseline_major)),
                        "intervention_is_outbreak": int(cf_summary["final_attack_rate"] >= args.outbreak_threshold),
                        "prevented_outbreak": int(bool(baseline_major and cf_summary["final_attack_rate"] < args.outbreak_threshold)),
                    }
                )
                rows.append(row)

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()

                if row["prevented_outbreak"]:
                    found_tipping = True
                    if args.stop_after_first_tipping:
                        break
            if found_tipping and args.stop_after_first_tipping:
                break

    if rows:
        ranked = sorted(rows, key=lambda r: (int(r["prevented_outbreak"]), float(r["attack_rate_reduction"])), reverse=True)
        write_dict_rows_csv(output_dir / "quarantine_counterfactuals_ranked.csv", ranked, fieldnames=list(ranked[0].keys()))

    summary = {
        "n_agents_requested": len(agent_ids),
        "n_days_requested": len(days),
        "n_counterfactuals_completed": len(rows),
        "search_skipped": False,
        "baseline": baseline_summary,
        "outbreak_threshold": float(args.outbreak_threshold),
        "outbreak_rf_gate": rf_gate_summary,
        "n_prevented_outbreak": int(sum(int(r["prevented_outbreak"]) for r in rows)),
        "best_attack_rate_reduction": float(max((float(r["attack_rate_reduction"]) for r in rows), default=0.0)),
    }
    (output_dir / "quarantine_search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote counterfactuals to: {csv_path}")


if __name__ == "__main__":
    main()
