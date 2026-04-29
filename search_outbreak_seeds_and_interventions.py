import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
from outbreak_ml_utils import DefaultOutbreakThreshold, write_dict_rows_csv
from single_agent_quarantine_search import _infer_outbreak_rf_settings, collect_baseline, make_cfg, metrics_summary, parse_selection, predict_baseline_outbreak_with_rf, run_counterfactual, write_metrics_csv


SeedStart = 0
MaxSeedAttempts = 500
TargetOutbreaks = 10

SusceptibilityMin = 1.302
SusceptibilityMax = 1.303
SimDays = 60

# "all", "0:100", or a comma list such as "3,15,27".
Agents = "all"

# Half-open interval: "0:40" means days 0 through 39.
Days = "0:40"

OutbreakThreshold = DefaultOutbreakThreshold  # usually 0.30
OutputDir = "threshold_seed_outbreak_search"

OutbreakRfModelDir = "rf_outbreak_model"
OutbreakRfModelFile = "outbreak_random_forest.joblib"

# Low value means conservative screening: fewer possible outbreaks are skipped.
OutbreakRfProbabilityThreshold = 0.10
OutbreakRfHistoryDays = None
OutbreakRfEndDay = None

# Set to True only if you want to ignore the RF screen and verify every seed.
SkipRfScreen = False

# For the question "can this outbreak be controlled by one intervention?",
# stop each outbreak case as soon as the first controlling intervention is found.
StopAfterFirstControllingIntervention = True


# -----------------------------------------------------------------------------
# IMPLEMENTATION
# -----------------------------------------------------------------------------


def make_args(**overrides):
    values = {
        "seed": SeedStart,
        "seed_start": SeedStart,
        "max_seed_attempts": MaxSeedAttempts,
        "target_outbreaks": TargetOutbreaks,
        "susceptibility_min": SusceptibilityMin,
        "susceptibility_max": SusceptibilityMax,
        "sim_days": SimDays,
        "agents": Agents,
        "days": Days,
        "outbreak_threshold": OutbreakThreshold,
        "output_dir": OutputDir,
        "outbreak_rf_model_dir": OutbreakRfModelDir,
        "outbreak_rf_model_file": OutbreakRfModelFile,
        "outbreak_rf_probability_threshold": OutbreakRfProbabilityThreshold,
        "outbreak_rf_history_days": OutbreakRfHistoryDays,
        "outbreak_rf_end_day": OutbreakRfEndDay,
        "skip_rf_screen": SkipRfScreen,
        "stop_after_first_controlling_intervention": StopAfterFirstControllingIntervention,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def clone_args(args, **overrides):
    values = vars(args).copy()
    values.update(overrides)
    return SimpleNamespace(**values)


def append_row_csv(path, row, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        if path.exists() and path.stat().st_size > 0:
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                fieldnames = next(reader)
        else:
            fieldnames = list(row.keys())

    new_keys = [key for key in row.keys() if key not in fieldnames]
    if new_keys and path.exists() and path.stat().st_size > 0:
        with path.open("r", newline="", encoding="utf-8") as f:
            old_rows = list(csv.DictReader(f))
        fieldnames = fieldnames + new_keys
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old in old_rows:
                writer.writerow({name: old.get(name, "") for name in fieldnames})
    else:
        fieldnames = fieldnames + new_keys

    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({name: row.get(name, "") for name in fieldnames})

    return fieldnames


def screen_seed_with_rf(args, seed):
    """Run an early baseline window and score it with the outbreak RF."""
    history_days, end_day = _infer_outbreak_rf_settings(Path(args.outbreak_rf_model_dir), args)
    early_sim_days = max(end_day + 1, history_days)
    early_args = clone_args(args, seed=seed, sim_days=min(int(args.sim_days), int(early_sim_days)))

    try:
        early_metrics, _ = collect_baseline(early_args)
        n_agents = int(make_cfg(early_args, do_quarantine=False).n_agents)
        rf_summary = predict_baseline_outbreak_with_rf(early_args, early_metrics, n_agents)
        return {
            "seed": int(seed),
            "rf_screen_error": "",
            "rf_predicted_outbreak": int(rf_summary["predicted_outbreak"]),
            "rf_probability": float(rf_summary["predicted_outbreak_probability"]),
            "rf_probability_threshold": float(rf_summary["probability_threshold"]),
            "rf_history_days": int(rf_summary["history_days"]),
            "rf_end_day": int(rf_summary["end_day"]),
            "rf_missing_features": int(rf_summary.get("n_missing_features", 0)),
            "rf_missing_koopman_features": int(rf_summary.get("n_missing_koopman_features", 0)),
        }
    except Exception as exc:
        return {
            "seed": int(seed),
            "rf_screen_error": str(exc),
            "rf_predicted_outbreak": 0,
            "rf_probability": 0.0,
            "rf_probability_threshold": float(args.outbreak_rf_probability_threshold),
        }


def run_one_outbreak_case(args, *, seed, case_index, case_dir):
    """Run full baseline. If it is an outbreak, search single-agent quarantines."""
    case_args = clone_args(args, seed=seed)
    n_agents = int(make_cfg(case_args, do_quarantine=False).n_agents)
    agent_ids = parse_selection(args.agents, n_agents)
    days = parse_selection(args.days, int(args.sim_days))

    case_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running full baseline for seed={seed}...")
    baseline_metrics, snapshots = collect_baseline(case_args)
    baseline_summary = metrics_summary(baseline_metrics, n_agents)
    baseline_is_outbreak = baseline_summary["final_attack_rate"] >= args.outbreak_threshold

    write_metrics_csv(case_dir / "baseline_metrics.csv", baseline_metrics)
    (case_dir / "baseline_summary.json").write_text(json.dumps(baseline_summary, indent=2), encoding="utf-8")

    summary = {
        "case_index": int(case_index),
        "seed": int(seed),
        "baseline_is_outbreak": int(bool(baseline_is_outbreak)),
        "baseline_final_attack_rate": float(baseline_summary["final_attack_rate"]),
        "baseline_peak_infected": int(baseline_summary["peak_infected"]),
        "baseline_peak_day": int(baseline_summary["peak_day"]),
        "outbreak_threshold": float(args.outbreak_threshold),
        "searched": 0,
        "controlled_by_single_intervention": 0,
        "n_counterfactuals_completed": 0,
        "n_prevented_outbreak": 0,
        "best_attack_rate_reduction": 0.0,
        "best_agent_id": "",
        "best_day": "",
    }

    if not baseline_is_outbreak:
        summary["skip_reason"] = "full_baseline_not_outbreak"
        (case_dir / "case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Seed {seed}: full baseline is not an outbreak " f"(attack rate={baseline_summary['final_attack_rate']:.4f}). Skipping interventions.")
        return summary

    rows = []
    csv_path = case_dir / "quarantine_counterfactuals.csv"
    total = len(agent_ids) * len(days)
    completed = 0
    writer = None
    found_control = False

    print(f"Searching interventions for verified outbreak seed={seed}: {total} candidates")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        for agent_id in agent_ids:
            for day in days:
                completed += 1
                snapshot = snapshots.get((agent_id, day))
                if snapshot is None:
                    continue

                print(f"[case {case_index} seed {seed}] [{completed}/{total}] quarantine agent={agent_id} day={day}")
                cf_metrics = run_counterfactual(case_args, int(agent_id), int(day))
                cf_summary = metrics_summary(cf_metrics, n_agents)

                row = dict(snapshot)
                row.update(
                    {
                        "case_index": int(case_index),
                        "seed": int(seed),
                        "susceptibility_min": float(args.susceptibility_min),
                        "susceptibility_max": float(args.susceptibility_max),
                        "sim_days": int(args.sim_days),
                        "baseline_final_attack_rate": float(baseline_summary["final_attack_rate"]),
                        "baseline_peak_infected": int(baseline_summary["peak_infected"]),
                        "baseline_peak_day": int(baseline_summary["peak_day"]),
                        "intervention_final_attack_rate": float(cf_summary["final_attack_rate"]),
                        "intervention_final_susceptible": int(cf_summary["final_susceptible"]),
                        "intervention_peak_infected": int(cf_summary["peak_infected"]),
                        "intervention_peak_day": int(cf_summary["peak_day"]),
                        "attack_rate_reduction": float(baseline_summary["final_attack_rate"] - cf_summary["final_attack_rate"]),
                        "baseline_is_outbreak": 1,
                        "intervention_is_outbreak": int(cf_summary["final_attack_rate"] >= args.outbreak_threshold),
                        "prevented_outbreak": int(cf_summary["final_attack_rate"] < args.outbreak_threshold),
                    }
                )
                rows.append(row)

                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    writer.writeheader()
                writer.writerow(row)
                f.flush()

                if int(row["prevented_outbreak"]):
                    found_control = True
                    if args.stop_after_first_controlling_intervention:
                        break

            if found_control and args.stop_after_first_controlling_intervention:
                break

    if rows:
        ranked = sorted(rows, key=lambda r: (int(r["prevented_outbreak"]), float(r["attack_rate_reduction"])), reverse=True)
        write_dict_rows_csv(case_dir / "quarantine_counterfactuals_ranked.csv", ranked, fieldnames=list(ranked[0].keys()))
        best = ranked[0]
        summary.update(
            {
                "best_attack_rate_reduction": float(best["attack_rate_reduction"]),
                "best_agent_id": int(best["agent_id"]),
                "best_day": int(best["day"]),
            }
        )

    summary.update(
        {
            "searched": 1,
            "controlled_by_single_intervention": int(any(int(r["prevented_outbreak"]) for r in rows)),
            "n_counterfactuals_completed": int(len(rows)),
            "n_prevented_outbreak": int(sum(int(r["prevented_outbreak"]) for r in rows)),
        }
    )
    (case_dir / "case_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main():
    args = make_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    screening_csv = output_dir / "seed_screening.csv"
    cases_csv = output_dir / "outbreak_case_summaries.csv"

    screening_fields = None
    case_fields = None
    verified_outbreaks = 0
    searched_outbreaks = 0
    controlled_outbreaks = 0
    last_seed_considered = None

    print("Experiment settings:")
    print(json.dumps(vars(args), indent=2, default=str))

    seed_stop = int(args.seed_start) + int(args.max_seed_attempts)
    for seed in range(int(args.seed_start), seed_stop):
        last_seed_considered = seed
        if verified_outbreaks >= int(args.target_outbreaks):
            break

        print(f"\n=== Screening seed {seed} ===")
        if args.skip_rf_screen:
            screen = {
                "seed": int(seed),
                "rf_screen_skipped": 1,
                "rf_predicted_outbreak": 1,
                "rf_probability": math.nan,
                "rf_probability_threshold": float(args.outbreak_rf_probability_threshold),
            }
        else:
            screen = screen_seed_with_rf(args, seed)
            screen["rf_screen_skipped"] = 0

        if not int(screen.get("rf_predicted_outbreak", 0)):
            screen["full_baseline_verified"] = 0
            screen["searched"] = 0
            screen["skip_reason"] = screen.get("rf_screen_error") or "rf_predicted_no_outbreak"
            screening_fields = append_row_csv(screening_csv, screen, screening_fields)
            print(f"Seed {seed}: RF screen did not pass " f"(P={float(screen.get('rf_probability', 0.0)):.4f}). Trying next seed.")
            continue

        case_index = verified_outbreaks + 1
        case_dir = output_dir / f"case_{case_index:02d}_seed_{seed}"
        case_summary = run_one_outbreak_case(args, seed=seed, case_index=case_index, case_dir=case_dir)

        screen.update(
            {
                "full_baseline_verified": 1,
                "baseline_is_outbreak": int(case_summary["baseline_is_outbreak"]),
                "baseline_final_attack_rate": float(case_summary["baseline_final_attack_rate"]),
                "baseline_peak_infected": int(case_summary["baseline_peak_infected"]),
                "baseline_peak_day": int(case_summary["baseline_peak_day"]),
                "searched": int(case_summary["searched"]),
                "controlled_by_single_intervention": int(case_summary["controlled_by_single_intervention"]),
                "n_counterfactuals_completed": int(case_summary["n_counterfactuals_completed"]),
                "n_prevented_outbreak": int(case_summary["n_prevented_outbreak"]),
                "best_attack_rate_reduction": float(case_summary["best_attack_rate_reduction"]),
                "case_dir": str(case_dir),
                "skip_reason": case_summary.get("skip_reason", ""),
            }
        )
        screening_fields = append_row_csv(screening_csv, screen, screening_fields)

        if int(case_summary["baseline_is_outbreak"]):
            verified_outbreaks += 1
            case_fields = append_row_csv(cases_csv, case_summary, case_fields)
            if int(case_summary["searched"]):
                searched_outbreaks += 1
            if int(case_summary["controlled_by_single_intervention"]):
                controlled_outbreaks += 1

    batch_summary = {
        "seed_start": int(args.seed_start),
        "max_seed_attempts": int(args.max_seed_attempts),
        "last_seed_considered": int(last_seed_considered) if last_seed_considered is not None else None,
        "target_outbreaks": int(args.target_outbreaks),
        "verified_outbreaks": int(verified_outbreaks),
        "searched_outbreaks": int(searched_outbreaks),
        "controlled_outbreaks": int(controlled_outbreaks),
        "all_verified_outbreaks_controlled": bool(verified_outbreaks > 0 and controlled_outbreaks == verified_outbreaks),
        "susceptibility_min": float(args.susceptibility_min),
        "susceptibility_max": float(args.susceptibility_max),
        "outbreak_threshold": float(args.outbreak_threshold),
        "outbreak_rf_probability_threshold": float(args.outbreak_rf_probability_threshold),
        "stop_after_first_controlling_intervention": bool(args.stop_after_first_controlling_intervention),
        "screening_csv": str(screening_csv),
        "outbreak_case_summaries_csv": str(cases_csv),
    }
    (output_dir / "batch_summary.json").write_text(json.dumps(batch_summary, indent=2), encoding="utf-8")

    print("\nBatch summary:")
    print(json.dumps(batch_summary, indent=2))


if __name__ == "__main__":
    main()
