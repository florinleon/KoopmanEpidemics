import csv
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn


DatasetPath = Path("baseline_dataset.npz")
TrainingDir = Path("koopman_training")
OutputDir = Path("koopman_rollout_indicators")

# Modes:
# - "early_first_window": one rollout per run, starting from the earliest training-style window.
# - "all_valid_windows": all valid start windows are rolled out; the printed prediction is the mean over starts.
Mode = "early_first_window"

# Used only in early_first_window mode.
EarlyReferenceEndDay = 4

# If True, peak day is reported with 0-based indexing. If False, add 1 for 1-based day numbering.
ZeroBasedDayIndex = True

# Output formatting.
AttackRateAsPercent = True
Decimals = 2

UseCudaIfAvailable = True

RequiredKeys = {"trajectories", "valid_lengths", "observable_names", "seeds", "peak_infected", "peak_day", 
                "final_susceptible", "final_attack_size", "sim_stopped_early", "config_json", "run_paths"}


class DeepKoopmanModel(nn.Module):
    def __init__(self, history_days, n_features, hidden_dim, latent_dim):
        super().__init__()
        input_dim = history_days * n_features
        self.history_days = history_days
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_features))
        self.attack_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())
        self.class_head = nn.Sequential(nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(), nn.Linear(hidden_dim // 2, 2))
        self.K = nn.Parameter(torch.eye(latent_dim))


    def encode(self, histories):
        flat = histories.reshape(histories.shape[0], -1)
        return self.encoder(flat)


    def rollout_observables(self, z0, horizon):
        preds = []
        z = z0
        for _ in range(horizon):
            z = z @ self.K.T
            preds.append(self.decoder(z))
        if not preds:
            return torch.empty((z0.shape[0], 0, self.n_features), dtype=z0.dtype, device=z0.device)
        return torch.stack(preds, dim=1)


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = RequiredKeys.difference(data.files)
        if missing:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def load_checkpoint(training_dir, device):
    ckpt_path = training_dir / "koopman_model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Unexpected checkpoint format. Expected a dict with a 'state_dict' key.")

    state_dict = checkpoint["state_dict"]
    history_days = int(checkpoint.get("history_days", 5))
    latent_dim = int(checkpoint.get("latent_dim", state_dict["K"].shape[0]))
    first_weight = state_dict["encoder.0.weight"]
    hidden_dim = int(first_weight.shape[0])
    input_dim = int(first_weight.shape[1])
    if input_dim % history_days != 0:
        raise ValueError("Encoder input dimension is not divisible by history_days.")
    n_features = input_dim // history_days

    model = DeepKoopmanModel(history_days=history_days, n_features=n_features, hidden_dim=hidden_dim, latent_dim=latent_dim)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint


def infer_n_agents(trajectory, valid_length, config_json_value):
    value = 0
    try:
        cfg = json.loads(str(config_json_value))
        value = int(cfg.get("n_agents", 0))
    except Exception:
        value = 0
    if value <= 0 and valid_length > 0:
        row = trajectory[0, :4]
        if np.all(np.isfinite(row)):
            value = int(np.round(float(np.sum(row))))
    if value <= 0:
        raise ValueError("Could not infer the number of agents for a trajectory.")
    return value


def get_observable_index(observable_names, name):
    names = [str(x) for x in observable_names]
    if name not in names:
        raise KeyError(f"Observable '{name}' not found. Available: {names}")
    return names.index(name)


def normalize_history(history, mean, std):
    return (history - mean[None, :]) / std[None, :]


def denormalize_observations(obs, mean, std):
    return obs * std[None, :] + mean[None, :]


def build_history(trajectory, end_day, history_days):
    start = end_day - history_days + 1
    return trajectory[start:end_day + 1, :]


def rollout_from_end_day(model, trajectory, end_day, mean, std, device):
    history = build_history(trajectory, end_day, model.history_days)
    norm_history = normalize_history(history, mean, std)
    horizon = trajectory.shape[0] - 1 - end_day

    with torch.no_grad():
        history_tensor = torch.from_numpy(norm_history.astype(np.float32)).unsqueeze(0).to(device)
        z0 = model.encode(history_tensor)
        future_norm = model.rollout_observables(z0, horizon).squeeze(0).cpu().numpy()

    if horizon > 0:
        future_obs = denormalize_observations(future_norm, mean, std)
        combined = np.concatenate([trajectory[:end_day + 1, :], future_obs], axis=0)
    else:
        combined = trajectory.copy()
    return combined


def compute_final_attack_rate(trajectory, susceptible_idx, n_agents):
    s_final = float(trajectory[-1, susceptible_idx])
    return 1.0 - (s_final / float(n_agents))


def compute_peak_day(trajectory, infected_idx):
    return float(int(np.argmax(trajectory[:, infected_idx])))


def format_attack_rate(value):
    scaled = value * 100.0 if AttackRateAsPercent else value
    return f"{scaled:.{Decimals}f}"


def format_day(value):
    day_value = value if ZeroBasedDayIndex else value + 1.0
    return f"{day_value:.{Decimals}f}"


def select_end_days(length, history_days, mode, early_reference_end_day):
    min_end_day = history_days - 1
    max_end_day = length - 1
    if max_end_day < min_end_day:
        return []

    if mode == "early_first_window":
        chosen = max(min_end_day, early_reference_end_day)
        if chosen > max_end_day:
            return []
        return [chosen]

    if mode == "all_valid_windows":
        # Require at least one future day so the rollout actually predicts something beyond the observed prefix.
        if max_end_day - 1 < min_end_day:
            return []
        return list(range(min_end_day, max_end_day))

    raise ValueError("Mode must be 'early_first_window' or 'all_valid_windows'.")


def summarize_run_predictions(predictions):
    attack_rates = np.array([p[0] for p in predictions], dtype=np.float64)
    peak_days = np.array([p[1] for p in predictions], dtype=np.float64)
    return float(np.mean(attack_rates)), float(np.mean(peak_days))


def main():
    device = torch.device("cuda" if UseCudaIfAvailable and torch.cuda.is_available() else "cpu")
    OutputDir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(DatasetPath)
    model, checkpoint = load_checkpoint(TrainingDir, device)

    observable_names = [str(x) for x in data["observable_names"]]
    susceptible_idx = get_observable_index(observable_names, "susceptible")
    infected_idx = get_observable_index(observable_names, "infected")

    mean = np.array(checkpoint.get("mean"), dtype=np.float32)
    std = np.array(checkpoint.get("std"), dtype=np.float32)
    if mean.shape[0] != len(observable_names) or std.shape[0] != len(observable_names):
        raise ValueError("Normalization statistics do not match the observable dimension.")

    trajectories = data["trajectories"]
    valid_lengths = data["valid_lengths"]
    seeds = data["seeds"]
    config_json = data["config_json"]

    lines = []
    rows = []

    header = f"Mode={Mode} | dataset={DatasetPath} | training_dir={TrainingDir} | history_days={model.history_days} | attack_rate_as_percent={AttackRateAsPercent}"
    lines.append(header)
    lines.append("-" * len(header))

    n_runs = trajectories.shape[0]
    for run_idx in range(n_runs):
        length = int(valid_lengths[run_idx])
        if length < model.history_days:
            continue

        true_traj = trajectories[run_idx, :length, :].astype(np.float32)
        n_agents = infer_n_agents(true_traj, length, config_json[run_idx])
        true_attack_rate = compute_final_attack_rate(true_traj, susceptible_idx, n_agents)
        true_peak_day = compute_peak_day(true_traj, infected_idx)

        end_days = select_end_days(length, model.history_days, Mode, EarlyReferenceEndDay)
        if not end_days:
            continue

        predictions = []
        for end_day in end_days:
            pred_traj = rollout_from_end_day(model, true_traj, end_day, mean, std, device)
            pred_attack_rate = compute_final_attack_rate(pred_traj, susceptible_idx, n_agents)
            pred_peak_day = compute_peak_day(pred_traj, infected_idx)
            predictions.append((pred_attack_rate, pred_peak_day))

        pred_attack_rate, pred_peak_day = summarize_run_predictions(predictions)

        line = f"run={run_idx:04d} seed={int(seeds[run_idx])} | attack_rate real={format_attack_rate(true_attack_rate)} vs pred={format_attack_rate(pred_attack_rate)} | peak_day real={format_day(true_peak_day)} vs pred={format_day(pred_peak_day)}"
        if Mode == "all_valid_windows":
            line += f" | starts={len(end_days)}"
        lines.append(line)

        rows.append({"run_index": int(run_idx), "seed": int(seeds[run_idx]), "n_agents": int(n_agents), "n_time_points": int(length), "mode": Mode, "n_start_windows": int(len(end_days)), "attack_rate_true": float(true_attack_rate), "attack_rate_pred": float(pred_attack_rate), "peak_day_true": float(true_peak_day if ZeroBasedDayIndex else true_peak_day + 1.0), "peak_day_pred": float(pred_peak_day if ZeroBasedDayIndex else pred_peak_day + 1.0)})

    txt_path = OutputDir / f"rollout_indicators_{Mode}.txt"
    csv_path = OutputDir / f"rollout_indicators_{Mode}.csv"
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_index", "seed", "n_agents", "n_time_points", "mode", "n_start_windows", "attack_rate_true", "attack_rate_pred", "peak_day_true", "peak_day_pred"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    for line in lines:
        print(line)
    print(f"\nWritten to:\n  {txt_path}\n  {csv_path}")


if __name__ == "__main__":
    main()
