import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


DatasetPath = Path("baseline_dataset.npz")
OutputDir = Path("koopman_training")

HistoryDays = 5
ForecastHorizon = 5
LatentDim = 6
HiddenDim = 128
BatchSize = 256
MaxEpochs = 100
LearningRate = 1e-3
WeightDecay = 1e-5
RandomSeed = 123

TrainFraction = 0.70
ValFraction = 0.15
TestFraction = 0.15

ContainmentThreshold = 0.10
MajorOutbreakThreshold = 0.30

# Use only early windows so the model focuses on the regime-decision phase,
# not on late-stage outbreak decay.
UseOnlyEarlyWindows = True
EarlyEndDayMin = HistoryDays - 1
EarlyEndDayMax = 12
# Optional cap to keep long runs from dominating if the early horizon is widened later.
MaxWindowsPerRun = None

PredictionWeight = 1.0
LinearityWeight = 0.25
AttackRegressionWeight = 0.20
ClassificationWeight = 0.20
GradClipNorm = 1.0

UseCudaIfAvailable = True
PlotResults = True
LatentTrajectoryRunsPerClass = 12

RequiredKeys = {
    "trajectories",
    "valid_lengths",
    "observable_names",
    "seeds",
    "peak_infected",
    "peak_day",
    "final_susceptible",
    "final_attack_size",
    "sim_stopped_early",
    "config_json",
    "run_paths",
}


@dataclass
class WindowedData:
    histories: np.ndarray
    future_obs: np.ndarray
    future_histories: np.ndarray
    attack_rates: np.ndarray
    class_labels: np.ndarray
    class_mask: np.ndarray
    run_indices: np.ndarray
    end_days: np.ndarray
    seeds: np.ndarray


class KoopmanWindowDataset(Dataset):
    def __init__(self, data):
        self.histories = torch.from_numpy(data.histories.astype(np.float32))
        self.future_obs = torch.from_numpy(data.future_obs.astype(np.float32))
        self.future_histories = torch.from_numpy(data.future_histories.astype(np.float32))
        self.attack_rates = torch.from_numpy(data.attack_rates.astype(np.float32))
        self.class_labels = torch.from_numpy(data.class_labels.astype(np.int64))
        self.class_mask = torch.from_numpy(data.class_mask.astype(np.bool_))
        self.run_indices = torch.from_numpy(data.run_indices.astype(np.int64))
        self.end_days = torch.from_numpy(data.end_days.astype(np.int64))
        self.seeds = torch.from_numpy(data.seeds.astype(np.int64))


    def __len__(self):
        return self.histories.shape[0]


    def __getitem__(self, idx):
        return {
            "history": self.histories[idx],
            "future_obs": self.future_obs[idx],
            "future_histories": self.future_histories[idx],
            "attack_rate": self.attack_rates[idx],
            "class_label": self.class_labels[idx],
            "class_mask": self.class_mask[idx],
            "run_index": self.run_indices[idx],
            "end_day": self.end_days[idx],
            "seed": self.seeds[idx],
        }


class DeepKoopmanModel(nn.Module):
    def __init__(self, history_days, n_features, hidden_dim, latent_dim, forecast_horizon):
        super().__init__()
        self.history_days = history_days
        self.n_features = n_features
        self.latent_dim = latent_dim
        self.forecast_horizon = forecast_horizon

        input_dim = history_days * n_features
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_features),
        )
        self.attack_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.class_head = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),
        )
        init = torch.eye(latent_dim) + 0.01 * torch.randn(latent_dim, latent_dim)
        self.K = nn.Parameter(init)


    def encode(self, histories):
        flat = histories.reshape(histories.shape[0], -1)
        return self.encoder(flat)


    def rollout(self, z0):
        latents = []
        preds = []
        z = z0
        for _ in range(self.forecast_horizon):
            z = z @ self.K.T
            latents.append(z)
            preds.append(self.decoder(z))
        return torch.stack(latents, dim=1), torch.stack(preds, dim=1)


    def forward(self, histories):
        z0 = self.encode(histories)
        future_latents, future_preds = self.rollout(z0)
        attack_rate_pred = self.attack_head(z0).squeeze(-1)
        class_logits = self.class_head(z0)
        return {
            "z0": z0,
            "future_latents": future_latents,
            "future_preds": future_preds,
            "attack_rate_pred": attack_rate_pred,
            "class_logits": class_logits,
        }


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = RequiredKeys.difference(data.files)
        if missing:
            raise KeyError(f"Dataset is missing required keys: {sorted(missing)}")
        return {key: data[key] for key in data.files}


def infer_n_agents(trajectories, valid_lengths, config_json):
    n_runs = trajectories.shape[0]
    n_agents = np.zeros(n_runs, dtype=np.int32)
    for i in range(n_runs):
        value = 0
        try:
            cfg = json.loads(str(config_json[i]))
            value = int(cfg.get("n_agents", 0))
        except Exception:
            value = 0
        if value <= 0 and int(valid_lengths[i]) > 0:
            row = trajectories[i, 0, :4]
            if np.all(np.isfinite(row)):
                value = int(np.round(float(np.sum(row))))
        if value <= 0:
            raise ValueError(f"Could not infer n_agents for run index {i}")
        n_agents[i] = value
    return n_agents


def split_run_indices(n_runs):
    indices = np.arange(n_runs)
    rng = np.random.default_rng(RandomSeed)
    rng.shuffle(indices)

    n_train = int(round(TrainFraction * n_runs))
    n_val = int(round(ValFraction * n_runs))
    n_train = min(max(n_train, 1), n_runs - 2)
    n_val = min(max(n_val, 1), n_runs - n_train - 1)
    n_test = n_runs - n_train - n_val
    if n_test <= 0:
        raise ValueError("Not enough runs to create train/val/test splits.")

    train_idx = np.sort(indices[:n_train])
    val_idx = np.sort(indices[n_train:n_train + n_val])
    test_idx = np.sort(indices[n_train + n_val:])
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def compute_normalization(trajectories, valid_lengths, run_indices):
    rows = []
    for run_idx in run_indices:
        length = int(valid_lengths[run_idx])
        if length <= 0:
            continue
        rows.append(trajectories[run_idx, :length, :])
    if not rows:
        raise ValueError("No valid rows available to compute normalization statistics.")
    stacked = np.concatenate(rows, axis=0)
    mean = np.nanmean(stacked, axis=0)
    std = np.nanstd(stacked, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_rows(rows, mean, std):
    return (rows - mean) / std


def build_windowed_data(trajectories, valid_lengths, seeds, attack_rates_per_run, run_indices, mean, std):
    histories = []
    future_obs = []
    future_histories = []
    attack_rates = []
    class_labels = []
    class_mask = []
    window_run_indices = []
    end_days = []
    window_seeds = []

    for run_idx in run_indices:
        length = int(valid_lengths[run_idx])
        if length < HistoryDays + ForecastHorizon:
            continue
        raw = trajectories[run_idx, :length, :]
        normalized = normalize_rows(raw, mean, std)
        attack_rate = float(attack_rates_per_run[run_idx])
        if attack_rate < ContainmentThreshold:
            label = 0
            label_mask = True
        elif attack_rate > MajorOutbreakThreshold:
            label = 1
            label_mask = True
        else:
            label = 0
            label_mask = False

        run_window_count = 0
        for end_day in range(HistoryDays - 1, length - ForecastHorizon):
            if UseOnlyEarlyWindows:
                if end_day < EarlyEndDayMin or end_day > EarlyEndDayMax:
                    continue
            if MaxWindowsPerRun is not None and run_window_count >= MaxWindowsPerRun:
                break

            history = normalized[end_day - HistoryDays + 1:end_day + 1, :]
            future = normalized[end_day + 1:end_day + 1 + ForecastHorizon, :]
            future_history_stack = []
            for step_ahead in range(1, ForecastHorizon + 1):
                hist = normalized[end_day - HistoryDays + 1 + step_ahead:end_day + 1 + step_ahead, :]
                future_history_stack.append(hist)

            histories.append(history)
            future_obs.append(future)
            future_histories.append(np.stack(future_history_stack, axis=0))
            attack_rates.append(attack_rate)
            class_labels.append(label)
            class_mask.append(label_mask)
            window_run_indices.append(int(run_idx))
            end_days.append(int(end_day))
            window_seeds.append(int(seeds[run_idx]))
            run_window_count += 1

    if not histories:
        raise ValueError("No valid windows were created. Check trajectory lengths or reduce the forecast horizon.")

    return WindowedData(
        histories=np.asarray(histories, dtype=np.float32),
        future_obs=np.asarray(future_obs, dtype=np.float32),
        future_histories=np.asarray(future_histories, dtype=np.float32),
        attack_rates=np.asarray(attack_rates, dtype=np.float32),
        class_labels=np.asarray(class_labels, dtype=np.int64),
        class_mask=np.asarray(class_mask, dtype=np.bool_),
        run_indices=np.asarray(window_run_indices, dtype=np.int64),
        end_days=np.asarray(end_days, dtype=np.int64),
        seeds=np.asarray(window_seeds, dtype=np.int64),
    )


def create_loader(data, shuffle):
    dataset = KoopmanWindowDataset(data)
    return DataLoader(dataset, batch_size=BatchSize, shuffle=shuffle, drop_last=False)


def compute_losses(model, batch, device):
    history = batch["history"].to(device)
    future_obs = batch["future_obs"].to(device)
    future_histories = batch["future_histories"].to(device)
    attack_rate = batch["attack_rate"].to(device)
    class_label = batch["class_label"].to(device)
    class_mask = batch["class_mask"].to(device)

    outputs = model(history)
    encoded_future_histories = model.encode(future_histories.reshape(-1, HistoryDays, history.shape[-1]))
    encoded_future_histories = encoded_future_histories.reshape(history.shape[0], ForecastHorizon, LatentDim)

    prediction_loss = torch.mean((outputs["future_preds"] - future_obs) ** 2)
    linearity_loss = torch.mean((outputs["future_latents"] - encoded_future_histories) ** 2)
    attack_loss = torch.mean((outputs["attack_rate_pred"] - attack_rate) ** 2)

    if torch.any(class_mask):
        classification_loss = nn.functional.cross_entropy(outputs["class_logits"][class_mask], class_label[class_mask])
        class_prob = torch.softmax(outputs["class_logits"][class_mask], dim=-1)[:, 1]
        class_pred = (class_prob >= 0.5).long()
        class_acc = float((class_pred == class_label[class_mask]).float().mean().detach().cpu())
    else:
        classification_loss = torch.tensor(0.0, device=device)
        class_acc = math.nan

    total_loss = PredictionWeight * prediction_loss + LinearityWeight * linearity_loss + AttackRegressionWeight * attack_loss + ClassificationWeight * classification_loss

    metrics = {
        "loss_total": float(total_loss.detach().cpu()),
        "loss_pred": float(prediction_loss.detach().cpu()),
        "loss_lin": float(linearity_loss.detach().cpu()),
        "loss_attack": float(attack_loss.detach().cpu()),
        "loss_class": float(classification_loss.detach().cpu()),
        "class_acc": class_acc,
    }
    return total_loss, metrics, outputs


def run_epoch(model, loader, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)

    sums = {}
    n_batches = 0
    for batch in loader:
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            total_loss, metrics, _ = compute_losses(model, batch, device)
            if is_train:
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), GradClipNorm)
                optimizer.step()
        n_batches += 1
        for key, value in metrics.items():
            if math.isnan(value):
                continue
            sums[key] = sums.get(key, 0.0) + value

    if n_batches == 0:
        raise ValueError("Empty loader.")
    return {key: value / n_batches for key, value in sums.items()}


def evaluate_and_collect(model, data, device):
    loader = create_loader(data, shuffle=False)
    model.eval()

    z0_all = []
    attack_rate_pred_all = []
    p_major_all = []
    run_indices_all = []
    end_days_all = []
    seeds_all = []
    attack_rate_true_all = []
    class_label_all = []
    class_mask_all = []

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch["history"].to(device))
            logits = outputs["class_logits"]
            probs = torch.softmax(logits, dim=-1)[:, 1]
            z0_all.append(outputs["z0"].cpu().numpy())
            attack_rate_pred_all.append(outputs["attack_rate_pred"].cpu().numpy())
            p_major_all.append(probs.cpu().numpy())
            run_indices_all.append(batch["run_index"].numpy())
            end_days_all.append(batch["end_day"].numpy())
            seeds_all.append(batch["seed"].numpy())
            attack_rate_true_all.append(batch["attack_rate"].numpy())
            class_label_all.append(batch["class_label"].numpy())
            class_mask_all.append(batch["class_mask"].numpy())

    z0 = np.concatenate(z0_all, axis=0)
    p_major = np.concatenate(p_major_all, axis=0)
    return {
        "z0": z0,
        "attack_rate_pred": np.concatenate(attack_rate_pred_all, axis=0),
        "p_major": p_major,
        "boundary_score": np.abs(p_major - 0.5),
        "run_index": np.concatenate(run_indices_all, axis=0),
        "end_day": np.concatenate(end_days_all, axis=0),
        "seed": np.concatenate(seeds_all, axis=0),
        "attack_rate_true": np.concatenate(attack_rate_true_all, axis=0),
        "class_label": np.concatenate(class_label_all, axis=0),
        "class_mask": np.concatenate(class_mask_all, axis=0),
    }


def save_training_curves(history, path):
    if plt is None:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    keys = [
        ("loss_total_train", "loss_total_val", "total loss"),
        ("loss_pred_train", "loss_pred_val", "prediction loss"),
        ("loss_attack_train", "loss_attack_val", "attack loss"),
        ("class_acc_train", "class_acc_val", "classification accuracy"),
    ]
    for ax, (train_key, val_key, title) in zip(axes.ravel(), keys):
        if train_key in history:
            ax.plot(history[train_key], label="train")
        if val_key in history:
            ax.plot(history[val_key], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def project_latent_to_2d(z0):
    if z0.shape[1] < 2:
        raise ValueError("Need at least 2 latent dimensions for 2D projection.")
    if PCA is None:
        return z0[:, :2]
    pca = PCA(n_components=2, random_state=RandomSeed)
    return pca.fit_transform(z0)


def compute_tsne_projection(z0):
    if z0.shape[0] < 3 or z0.shape[1] < 2 or TSNE is None:
        return None
    perplexity = min(30.0, max(5.0, float((z0.shape[0] - 1) // 3)))
    perplexity = min(perplexity, float(z0.shape[0] - 1))
    tsne = TSNE(n_components=2, random_state=RandomSeed, init="pca", learning_rate="auto", perplexity=perplexity, max_iter=1000)
    return tsne.fit_transform(z0)


def save_latent_scatter_by_class(outputs, path):
    if plt is None:
        return
    z0 = outputs["z0"]
    if z0.shape[1] < 2:
        return
    proj = project_latent_to_2d(z0)
    class_mask = outputs["class_mask"].astype(bool)
    labels = outputs["class_label"]

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    for label_value, label_name in [(0, "contained"), (1, "major")]:
        mask = class_mask & (labels == label_value)
        if np.any(mask):
            ax.scatter(proj[mask, 0], proj[mask, 1], s=12, label=label_name, alpha=0.7)
    boundary_mask = ~class_mask
    if np.any(boundary_mask):
        ax.scatter(proj[boundary_mask, 0], proj[boundary_mask, 1], s=12, label="boundary", alpha=0.7)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Latent space by class (PCA projection)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_latent_scatter_by_attack_rate(outputs, path):
    if plt is None:
        return
    z0 = outputs["z0"]
    if z0.shape[1] < 2:
        return
    proj = project_latent_to_2d(z0)
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=outputs["attack_rate_true"], s=12, alpha=0.75)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Latent space colored by final attack rate (PCA projection)")
    fig.colorbar(sc, ax=ax, label="final attack rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_boundary_score_vs_day(outputs, path):
    if plt is None:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = outputs["class_label"]
    class_mask = outputs["class_mask"].astype(bool)
    end_days = outputs["end_day"]
    p_major = outputs["p_major"]
    boundary_score = outputs["boundary_score"]

    for label_value, label_name in [(0, "contained"), (1, "major")]:
        mask = class_mask & (labels == label_value)
        if np.any(mask):
            axes[0].scatter(end_days[mask], p_major[mask], s=12, alpha=0.6, label=label_name)
            axes[1].scatter(end_days[mask], boundary_score[mask], s=12, alpha=0.6, label=label_name)
    axes[0].set_xlabel("end day of 5-day history")
    axes[0].set_ylabel("predicted probability of major outbreak")
    axes[0].set_title("Early-window outbreak score")
    axes[0].legend()
    axes[1].set_xlabel("end day of 5-day history")
    axes[1].set_ylabel("|p_major - 0.5|")
    axes[1].set_title("Boundary-distance score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_latent_trajectory_panels(outputs, path):
    if plt is None:
        return
    z0 = outputs["z0"]
    if z0.shape[1] < 2:
        return
    proj = project_latent_to_2d(z0)

    run_indices = outputs["run_index"]
    end_days = outputs["end_day"]
    class_labels = outputs["class_label"]
    class_mask = outputs["class_mask"].astype(bool)

    unique_runs = np.unique(run_indices)
    run_label = {}
    for run_idx in unique_runs:
        mask = run_indices == run_idx
        run_label[int(run_idx)] = int(class_labels[mask][0])

    contained_runs = [r for r in unique_runs if class_mask[run_indices == r][0] and run_label[int(r)] == 0]
    major_runs = [r for r in unique_runs if class_mask[run_indices == r][0] and run_label[int(r)] == 1]

    contained_runs = list(contained_runs[:LatentTrajectoryRunsPerClass])
    major_runs = list(major_runs[:LatentTrajectoryRunsPerClass])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for ax, runs, title in [(axes[0], contained_runs, "Contained runs"), (axes[1], major_runs, "Major-outbreak runs")]:
        for run_idx in runs:
            mask = run_indices == run_idx
            order = np.argsort(end_days[mask])
            z = proj[mask][order]
            days = end_days[mask][order]
            ax.plot(z[:, 0], z[:, 1], alpha=0.7)
            ax.scatter(z[0, 0], z[0, 1], s=18)
            ax.text(z[-1, 0], z[-1, 1], f"r{int(run_idx)} d{int(days[-1])}", fontsize=7)
        ax.set_title(title)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
    fig.suptitle("Early-window latent trajectories (PCA projection)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_latent_tsne_summary(outputs, path):
    if plt is None:
        return
    z0 = outputs["z0"]
    proj = compute_tsne_projection(z0)
    if proj is None:
        return

    class_mask = outputs["class_mask"].astype(bool)
    labels = outputs["class_label"]
    attack_rate = outputs["attack_rate_true"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for label_value, label_name in [(0, "contained"), (1, "major")]:
        mask = class_mask & (labels == label_value)
        if np.any(mask):
            ax.scatter(proj[mask, 0], proj[mask, 1], s=12, label=label_name, alpha=0.7)
    boundary_mask = ~class_mask
    if np.any(boundary_mask):
        ax.scatter(proj[boundary_mask, 0], proj[boundary_mask, 1], s=12, label="boundary", alpha=0.7)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Latent space by class (t-SNE)")
    ax.legend()

    ax = axes[1]
    sc = ax.scatter(proj[:, 0], proj[:, 1], c=attack_rate, s=12, alpha=0.75)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Latent space by final attack rate (t-SNE)")
    fig.colorbar(sc, ax=ax, label="final attack rate")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_run_level_summary(outputs):
    rows = []
    run_indices = outputs["run_index"]
    unique_runs = np.unique(run_indices)
    for run_idx in unique_runs:
        mask = run_indices == run_idx
        order = np.argsort(outputs["end_day"][mask])
        p_major = outputs["p_major"][mask][order]
        boundary_score = outputs["boundary_score"][mask][order]
        end_days = outputs["end_day"][mask][order]
        attack_rate_true = outputs["attack_rate_true"][mask][0]
        rows.append(
            {
                "run_index": int(run_idx),
                "seed": int(outputs["seed"][mask][0]),
                "class_label": int(outputs["class_label"][mask][0]),
                "class_mask": bool(outputs["class_mask"][mask][0]),
                "attack_rate_true": float(attack_rate_true),
                "n_windows": int(np.sum(mask)),
                "first_end_day": int(end_days[0]),
                "last_end_day": int(end_days[-1]),
                "p_major_first": float(p_major[0]),
                "p_major_last": float(p_major[-1]),
                "p_major_mean": float(np.mean(p_major)),
                "boundary_score_min": float(np.min(boundary_score)),
                "boundary_score_mean": float(np.mean(boundary_score)),
            }
        )
    rows.sort(key=lambda x: (x["class_label"], x["attack_rate_true"], x["seed"], x["run_index"]))
    return rows


def save_run_level_summary_csv(rows, path):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    set_seed(RandomSeed)
    OutputDir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if UseCudaIfAvailable and torch.cuda.is_available() else "cpu")
    dataset = load_dataset(DatasetPath)

    trajectories = np.asarray(dataset["trajectories"], dtype=np.float32)
    valid_lengths = np.asarray(dataset["valid_lengths"], dtype=np.int32)
    observable_names = [str(x) for x in dataset["observable_names"]]
    seeds = np.asarray(dataset["seeds"], dtype=np.int32)
    final_attack_size = np.asarray(dataset["final_attack_size"], dtype=np.int32)
    config_json = np.asarray(dataset["config_json"])

    n_runs, _, n_features = trajectories.shape
    if n_features != len(observable_names):
        raise ValueError("Observable name count does not match feature dimension.")

    n_agents = infer_n_agents(trajectories, valid_lengths, config_json)
    attack_rates_per_run = final_attack_size / n_agents
    run_splits = split_run_indices(n_runs)
    mean, std = compute_normalization(trajectories, valid_lengths, run_splits["train"])

    (OutputDir / "normalization_stats.json").write_text(
        json.dumps(
            {
                "observable_names": observable_names,
                "mean": mean.tolist(),
                "std": std.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    train_data = build_windowed_data(trajectories, valid_lengths, seeds, attack_rates_per_run, run_splits["train"], mean, std)
    val_data = build_windowed_data(trajectories, valid_lengths, seeds, attack_rates_per_run, run_splits["val"], mean, std)
    test_data = build_windowed_data(trajectories, valid_lengths, seeds, attack_rates_per_run, run_splits["test"], mean, std)

    train_loader = create_loader(train_data, shuffle=True)
    val_loader = create_loader(val_data, shuffle=False)

    model = DeepKoopmanModel(HistoryDays, n_features, HiddenDim, LatentDim, ForecastHorizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LearningRate, weight_decay=WeightDecay)

    history = {}
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(1, MaxEpochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)

        for key, value in train_metrics.items():
            history.setdefault(f"{key}_train", []).append(value)
        for key, value in val_metrics.items():
            history.setdefault(f"{key}_val", []).append(value)

        val_total = val_metrics["loss_total"]
        if val_total < best_val:
            best_val = val_total
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch:03d} | " f"train total={train_metrics['loss_total']:.5f} pred={train_metrics['loss_pred']:.5f} " 
              f"val total={val_metrics['loss_total']:.5f} pred={val_metrics['loss_pred']:.5f}")

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    checkpoint = {
        "state_dict": model.state_dict(),
        "observable_names": observable_names,
        "history_days": HistoryDays,
        "forecast_horizon": ForecastHorizon,
        "latent_dim": LatentDim,
        "mean": mean,
        "std": std,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }
    torch.save(checkpoint, OutputDir / "koopman_model.pt")

    train_outputs = evaluate_and_collect(model, train_data, device)
    val_outputs = evaluate_and_collect(model, val_data, device)
    test_outputs = evaluate_and_collect(model, test_data, device)

    np.savez_compressed(OutputDir / "latent_outputs_train.npz", **train_outputs)
    np.savez_compressed(OutputDir / "latent_outputs_val.npz", **val_outputs)
    np.savez_compressed(OutputDir / "latent_outputs_test.npz", **test_outputs)

    train_run_summary = build_run_level_summary(train_outputs)
    val_run_summary = build_run_level_summary(val_outputs)
    test_run_summary = build_run_level_summary(test_outputs)
    save_run_level_summary_csv(train_run_summary, OutputDir / "run_level_summary_train.csv")
    save_run_level_summary_csv(val_run_summary, OutputDir / "run_level_summary_val.csv")
    save_run_level_summary_csv(test_run_summary, OutputDir / "run_level_summary_test.csv")

    metrics_summary = {
        "dataset_path": str(DatasetPath.resolve()),
        "device": str(device),
        "n_runs": int(n_runs),
        "n_train_runs": int(len(run_splits["train"])),
        "n_val_runs": int(len(run_splits["val"])),
        "n_test_runs": int(len(run_splits["test"])),
        "n_train_windows": int(train_data.histories.shape[0]),
        "n_val_windows": int(val_data.histories.shape[0]),
        "n_test_windows": int(test_data.histories.shape[0]),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "observable_names": observable_names,
        "history_days": HistoryDays,
        "forecast_horizon": ForecastHorizon,
        "latent_dim": LatentDim,
        "containment_threshold": ContainmentThreshold,
        "major_outbreak_threshold": MajorOutbreakThreshold,
        "use_only_early_windows": UseOnlyEarlyWindows,
        "early_end_day_min": int(EarlyEndDayMin),
        "early_end_day_max": int(EarlyEndDayMax),
        "max_windows_per_run": MaxWindowsPerRun,
    }
    (OutputDir / "training_summary.json").write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
    (OutputDir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if PlotResults:
        save_training_curves(history, OutputDir / "training_curves.png")
        save_latent_scatter_by_class(test_outputs, OutputDir / "latent_scatter_test_by_class.png")
        save_latent_scatter_by_attack_rate(test_outputs, OutputDir / "latent_scatter_test_by_attack_rate.png")
        save_boundary_score_vs_day(test_outputs, OutputDir / "boundary_score_vs_day_test.png")
        save_latent_trajectory_panels(test_outputs, OutputDir / "latent_trajectories_test.png")
        save_latent_tsne_summary(test_outputs, OutputDir / "latent_tsne_test.png")

    print(f"Training complete. Outputs written to: {OutputDir.resolve()}")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Train/val/test windows: {train_data.histories.shape[0]} / {val_data.histories.shape[0]} / {test_data.histories.shape[0]}")
    print(f"Early-window mode: {UseOnlyEarlyWindows} | end_day in [{EarlyEndDayMin}, {EarlyEndDayMax}]")


if __name__ == "__main__":
    main()
