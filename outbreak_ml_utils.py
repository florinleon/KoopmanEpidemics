import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from config import default_config
from koopman_io import config_to_dict


DefaultOutbreakThreshold = 0.30
StateToCode = {"susceptible": 0, "infected": 1, "recovered": 2, "dead": 3}
ImmunityLevels = ("strong", "medium", "low", "compromised")

RequiredDatasetKeys = {
    "trajectories",
    "valid_lengths",
    "observable_names",
    "seeds",
    "final_attack_size",
    "config_json",
}


def make_config_clone(**overrides):
    """Return an instance-style config object.

    The original project stores defaults as class attributes. A clone avoids
    accidental cross-run mutation when many counterfactual simulations are run
    in one process.
    """
    base = config_to_dict(default_config())
    base.update(overrides)
    return SimpleNamespace(**base)


def load_npz_dataset(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = RequiredDatasetKeys.difference(data.files)
        if missing:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def parse_config_json(value):
    try:
        return json.loads(str(value))
    except Exception:
        return {}


def infer_n_agents_for_run(trajectory, valid_length, config_json_value):
    cfg = parse_config_json(config_json_value)
    n_agents = int(cfg.get("n_agents", 0) or 0)
    if n_agents <= 0 and valid_length > 0:
        first_row = np.asarray(trajectory[0, :4], dtype=np.float64)
        if np.all(np.isfinite(first_row)):
            n_agents = int(round(float(first_row.sum())))
    if n_agents <= 0:
        raise ValueError("Could not infer n_agents for one run.")
    return n_agents


def infer_n_agents(data):
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int32)
    config_json = np.asarray(data["config_json"])
    values = []
    for run_idx in range(trajectories.shape[0]):
        values.append(infer_n_agents_for_run(trajectories[run_idx], int(valid_lengths[run_idx]), config_json[run_idx]))
    return np.asarray(values, dtype=np.int32)


def final_attack_rates(data, n_agents=None):
    if n_agents is None:
        n_agents = infer_n_agents(data)
    final_attack_size = np.asarray(data["final_attack_size"], dtype=np.float64)
    return final_attack_size / n_agents.astype(np.float64)


def outbreak_labels(attack_rates, threshold=DefaultOutbreakThreshold):
    return (np.asarray(attack_rates, dtype=np.float64) >= float(threshold)).astype(np.int64)


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


def load_koopman_latent_features(training_dir):
    """Load latent Koopman outputs keyed by (run_index, end_day)."""
    training_dir = Path(training_dir)
    result = {}
    for split in ("train", "val", "test"):
        path = training_dir / f"latent_outputs_{split}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as data:
            run_indices = np.asarray(data["run_index"], dtype=np.int64)
            end_days = np.asarray(data["end_day"], dtype=np.int64)
            z0 = np.asarray(data["z0"], dtype=np.float64)
            p_major = np.asarray(data.get("p_major", np.zeros(len(run_indices))), dtype=np.float64)
            attack_rate_pred = np.asarray(data.get("attack_rate_pred", np.zeros(len(run_indices))), dtype=np.float64)
            boundary_score = np.asarray(data.get("boundary_score", np.zeros(len(run_indices))), dtype=np.float64)
            for idx in range(len(run_indices)):
                key = (int(run_indices[idx]), int(end_days[idx]))
                features = {f"koopman_z{j}": float(z0[idx, j]) for j in range(z0.shape[1])}
                features["koopman_p_major"] = float(p_major[idx])
                features["koopman_attack_rate_pred"] = float(attack_rate_pred[idx])
                features["koopman_boundary_score"] = float(boundary_score[idx])
                features["koopman_available"] = 1.0
                result[key] = features
    return result


def build_outbreak_window_table(data, *, history_days=5, end_day_min=None, end_day_max=12, 
    outbreak_threshold=DefaultOutbreakThreshold, koopman_training_dir=None, require_koopman=False):
    """Create window-level features for early outbreak classification.

    Each row represents one run observed through one end-of-history day. The
    label is the final outbreak status of the full unperturbed trajectory.
    """
    trajectories = np.asarray(data["trajectories"], dtype=np.float32)
    valid_lengths = np.asarray(data["valid_lengths"], dtype=np.int32)
    observable_names = [str(x) for x in data["observable_names"]]
    seeds = np.asarray(data["seeds"], dtype=np.int64)
    config_json = np.asarray(data["config_json"])

    n_agents = infer_n_agents(data)
    rates = final_attack_rates(data, n_agents)
    labels = outbreak_labels(rates, threshold=outbreak_threshold)
    latent_by_key = load_koopman_latent_features(koopman_training_dir) if koopman_training_dir else {}

    if end_day_min is None:
        end_day_min = history_days - 1

    records = []
    feature_dicts = []

    for run_idx in range(trajectories.shape[0]):
        length = int(valid_lengths[run_idx])
        if length < history_days:
            continue
        cfg = parse_config_json(config_json[run_idx])
        max_end = min(int(end_day_max), length - 1)
        for end_day in range(int(end_day_min), max_end + 1):
            start = end_day - history_days + 1
            if start < 0:
                continue
            latent = latent_by_key.get((run_idx, end_day), None)
            if require_koopman and latent is None:
                continue

            window = trajectories[run_idx, start:end_day + 1, :].astype(np.float64)
            features = {
                "run_index_numeric": float(run_idx),
                "seed_numeric": float(seeds[run_idx]),
                "end_day": float(end_day),
                "history_days": float(history_days),
                "susceptibility_min": float(cfg.get("susceptibility_min", math.nan)),
                "susceptibility_max": float(cfg.get("susceptibility_max", math.nan)),
            }
            for col_idx, name in enumerate(observable_names):
                _add_series_features(features, name, window[:, col_idx], int(n_agents[run_idx]))

            name_to_idx = {name: idx for idx, name in enumerate(observable_names)}
            if "infected" in name_to_idx and "susceptible" in name_to_idx:
                infected_last = float(window[-1, name_to_idx["infected"]])
                susceptible_last = float(window[-1, name_to_idx["susceptible"]])
                features["infected_to_susceptible_last"] = _ratio(infected_last, susceptible_last)
            if "new_infections" in name_to_idx and "infected" in name_to_idx:
                new_last = float(window[-1, name_to_idx["new_infections"]])
                infected_last = float(window[-1, name_to_idx["infected"]])
                features["new_infections_to_infected_last"] = _ratio(new_last, infected_last)
            if "infected_mobile" in name_to_idx and "infected" in name_to_idx:
                mobile_last = float(window[-1, name_to_idx["infected_mobile"]])
                infected_last = float(window[-1, name_to_idx["infected"]])
                features["infected_mobile_share_last"] = _ratio(mobile_last, infected_last)

            if latent is not None:
                features.update(latent)
            elif koopman_training_dir is not None:
                features["koopman_available"] = 0.0

            feature_dicts.append(features)
            records.append({"run_index": int(run_idx), "seed": int(seeds[run_idx]), "end_day": int(end_day), 
                            "final_attack_rate": float(rates[run_idx]), "outbreak_label": int(labels[run_idx]), 
                            "n_agents": int(n_agents[run_idx])})

    if not feature_dicts:
        raise ValueError("No feature rows were created. Check history/end-day settings.")

    feature_names = sorted({key for d in feature_dicts for key in d.keys()})
    try:
        import pandas as pd

        frame = pd.DataFrame(feature_dicts)
        frame = frame.reindex(columns=feature_names, fill_value=0.0)
        frame = frame.replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X = frame.to_numpy(dtype=np.float32)
    except Exception:
        X = np.zeros((len(feature_dicts), len(feature_names)), dtype=np.float32)
        for row_idx, features in enumerate(feature_dicts):
            for col_idx, name in enumerate(feature_names):
                value = features.get(name, 0.0)
                if value is None or not np.isfinite(value):
                    value = 0.0
                X[row_idx, col_idx] = float(value)
    y = np.asarray([r["outbreak_label"] for r in records], dtype=np.int64)
    return X, y, records, feature_names


def write_dict_rows_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def route_features(routine):
    arr = np.asarray(routine, dtype=np.float64)
    if arr.size == 0:
        return {
            "route_centroid_x": 0.0,
            "route_centroid_y": 0.0,
            "route_radius_mean": 0.0,
            "route_radius_max": 0.0,
            "route_step_distance_mean": 0.0,
            "route_step_distance_max": 0.0,
        }
    centroid = arr.mean(axis=0)
    radius = np.sqrt(((arr - centroid[None, :]) ** 2).sum(axis=1))
    if arr.shape[0] > 1:
        diffs = np.diff(np.vstack([arr, arr[0:1]]), axis=0)
        step_dist = np.sqrt((diffs ** 2).sum(axis=1))
    else:
        step_dist = np.zeros(1)
    return {
        "route_centroid_x": float(centroid[0]),
        "route_centroid_y": float(centroid[1]),
        "route_radius_mean": float(radius.mean()),
        "route_radius_max": float(radius.max()),
        "route_step_distance_mean": float(step_dist.mean()),
        "route_step_distance_max": float(step_dist.max()),
    }


def agent_snapshot_features(agent, day):
    features = {
        "agent_id": int(agent.agent_id),
        "day": int(day),
        "home_id": int(agent.home_id),
        "phase": int(agent.phase),
        "susceptibility": float(agent.susceptibility),
        "immunity_strength": str(agent.immunity_strength),
        "state": str(agent.state),
        "state_code": int(StateToCode.get(str(agent.state), -1)),
        "is_mobile": int(bool(agent.is_mobile)),
        "viral_load": float(agent.viral_load),
        "infection_day": -1 if agent.infection_day is None else int(agent.infection_day),
        "days_since_infection": -1 if agent.infection_day is None else int(day - agent.infection_day),
    }
    for level in ImmunityLevels:
        features[f"immunity_is_{level}"] = int(agent.immunity_strength == level)
    features.update(route_features(agent.routine))
    return features
