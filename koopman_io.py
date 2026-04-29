import json
from pathlib import Path
import numpy as np


ObservableNames = ("susceptible", "infected", "recovered", "dead", "new_infections", 
                   "new_recoveries", "new_deaths", "infected_mobile", "infected_homebound")


def _is_public_config_name(name):
    return not name.startswith("_") and name != "default_config"


def config_to_dict(cfg):
    # Support both class-style and instance-style config objects.
    result = {}
    for name in dir(cfg):
        if not _is_public_config_name(name):
            continue
        value = getattr(cfg, name)
        if callable(value):
            continue
        result[name] = value
    return result


def metrics_to_daily_observables(metrics):
    columns = [np.asarray(metrics[name], dtype=np.int32) for name in ObservableNames]
    if not columns:
        return np.zeros((0, 0), dtype=np.int32)
    return np.column_stack(columns)


def summarize_metrics(metrics, cfg):
    infected = np.asarray(metrics["infected"], dtype=np.int32)
    susceptible = np.asarray(metrics["susceptible"], dtype=np.int32)
    n_days = int(len(infected))

    if n_days == 0:
        return {"n_days_recorded": 0, "peak_infected": 0, "peak_day": -1, 
                "final_susceptible": int(getattr(cfg, "n_agents", 0)), "final_attack_size": 0, 
                "sim_stopped_early": False}

    peak_day = int(np.argmax(infected))
    final_susceptible = int(susceptible[-1])
    n_agents = int(getattr(cfg, "n_agents", 0))
    final_attack_size = int(n_agents - final_susceptible)
    sim_stopped_early = n_days < int(getattr(cfg, "sim_days", n_days))

    return {"n_days_recorded": n_days, "peak_infected": int(infected[peak_day]), 
            "peak_day": peak_day, "final_susceptible": final_susceptible, 
            "final_attack_size": final_attack_size, "sim_stopped_early": sim_stopped_early}


def save_run_npz(path, metrics, cfg, **extra_fields):
    path = Path(path)
    daily_observables = metrics_to_daily_observables(metrics)
    summary = summarize_metrics(metrics, cfg)
    payload = {"daily_observables": daily_observables, "observable_names": np.asarray(ObservableNames), 
               "day_index": np.arange(daily_observables.shape[0], dtype=np.int32), 
               "seed": np.int32(getattr(cfg, "seed", -1)), "config_json": json.dumps(config_to_dict(cfg), sort_keys=True), 
               "n_days_recorded": np.int32(summary["n_days_recorded"]), "peak_infected": np.int32(summary["peak_infected"]), 
               "peak_day": np.int32(summary["peak_day"]), "final_susceptible": np.int32(summary["final_susceptible"]), 
               "final_attack_size": np.int32(summary["final_attack_size"]), "sim_stopped_early": np.bool_(summary["sim_stopped_early"])}

    for key, value in extra_fields.items():
        if isinstance(value, (str, bytes)):
            payload[key] = value
        elif isinstance(value, (bool, np.bool_)):
            payload[key] = np.bool_(value)
        elif isinstance(value, (int, np.integer)):
            payload[key] = np.int64(value)
        elif isinstance(value, (float, np.floating)):
            payload[key] = np.float64(value)
        else:
            payload[key] = json.dumps(value, sort_keys=True)

    np.savez_compressed(path, **payload)
    return path


def build_batch_npz(run_paths, output_path):
    run_paths = [Path(p) for p in run_paths]
    if not run_paths:
        raise ValueError("No run files provided.")

    records = []
    max_days = 0
    n_features = None
    observable_names = None

    for run_path in run_paths:
        with np.load(run_path, allow_pickle=False) as data:
            daily = np.asarray(data["daily_observables"], dtype=np.float32)
            names = np.asarray(data["observable_names"])
            if n_features is None:
                n_features = int(daily.shape[1])
                observable_names = names
            elif int(daily.shape[1]) != n_features or not np.array_equal(names, observable_names):
                raise ValueError(f"Incompatible observable layout in {run_path}")

            records.append({"run_path": str(run_path), "daily_observables": daily, "seed": int(data["seed"]), 
                            "peak_infected": int(data["peak_infected"]), "peak_day": int(data["peak_day"]), 
                            "final_susceptible": int(data["final_susceptible"]), "final_attack_size": int(data["final_attack_size"]), 
                            "sim_stopped_early": bool(data["sim_stopped_early"]), "n_days_recorded": int(data["n_days_recorded"]), 
                            "config_json": str(data["config_json"])})
            max_days = max(max_days, daily.shape[0])

    trajectories = np.full((len(records), max_days, n_features), np.nan, dtype=np.float32)
    valid_lengths = np.zeros(len(records), dtype=np.int32)
    seeds = np.zeros(len(records), dtype=np.int32)
    peak_infected = np.zeros(len(records), dtype=np.int32)
    peak_day = np.zeros(len(records), dtype=np.int32)
    final_susceptible = np.zeros(len(records), dtype=np.int32)
    final_attack_size = np.zeros(len(records), dtype=np.int32)
    sim_stopped_early = np.zeros(len(records), dtype=np.bool_)
    config_json = np.empty(len(records), dtype=f"<U{max(len(r['config_json']) for r in records)}")
    run_paths_arr = np.empty(len(records), dtype=f"<U{max(len(r['run_path']) for r in records)}")

    for idx, record in enumerate(records):
        length = record["daily_observables"].shape[0]
        trajectories[idx, :length, :] = record["daily_observables"]
        valid_lengths[idx] = length
        seeds[idx] = record["seed"]
        peak_infected[idx] = record["peak_infected"]
        peak_day[idx] = record["peak_day"]
        final_susceptible[idx] = record["final_susceptible"]
        final_attack_size[idx] = record["final_attack_size"]
        sim_stopped_early[idx] = record["sim_stopped_early"]
        config_json[idx] = record["config_json"]
        run_paths_arr[idx] = record["run_path"]

    output_path = Path(output_path)
    np.savez_compressed(output_path, trajectories=trajectories, valid_lengths=valid_lengths, observable_names=observable_names, 
                        seeds=seeds, peak_infected=peak_infected, peak_day=peak_day, final_susceptible=final_susceptible, 
                        final_attack_size=final_attack_size, sim_stopped_early=sim_stopped_early, config_json=config_json, run_paths=run_paths_arr)
    return output_path
