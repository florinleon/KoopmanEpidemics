import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


DatasetPath = Path("baseline_dataset.npz")
OutputDir = Path("koopman_inspection")
ContainmentThreshold = 0.10
MajorOutbreakThreshold = 0.30
MaxExampleRuns = 6

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


def load_dataset(path):
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        missing = RequiredKeys.difference(data.files)
        if missing:
            missing_txt = ", ".join(sorted(missing))
            raise KeyError(f"Dataset is missing required keys: {missing_txt}")
        return {key: data[key] for key in data.files}


def infer_n_agents(trajectories, valid_lengths, config_json):
    n_runs = trajectories.shape[0]
    n_agents = np.zeros(n_runs, dtype=np.int32)
    for i in range(n_runs):
        value = None
        try:
            cfg = json.loads(str(config_json[i]))
            value = int(cfg.get("n_agents", 0))
        except Exception:
            value = 0
        if value <= 0:
            length = int(valid_lengths[i])
            if length > 0:
                first_row = trajectories[i, 0, :4]
                if np.all(np.isfinite(first_row)):
                    value = int(np.round(float(np.sum(first_row))))
        if value <= 0:
            raise ValueError(f"Could not infer n_agents for run index {i}")
        n_agents[i] = value
    return n_agents


def attack_rate_labels(attack_rates):
    labels = np.full(attack_rates.shape, "boundary", dtype=object)
    labels[attack_rates < ContainmentThreshold] = "contained"
    labels[attack_rates > MajorOutbreakThreshold] = "major_outbreak"
    return labels


def summary_stats(values):
    if values.size == 0:
        return {"min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def write_text_summary(path, content):
    path.write_text(content, encoding="utf-8")


def save_histogram(values, title, xlabel, path):
    if plt is None:
        return
    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(values, bins=20)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_example_trajectories(trajectories, valid_lengths, observable_names, labels, seeds, path):
    if plt is None:
        return
    infected_idx = observable_names.index("infected")
    new_inf_idx = observable_names.index("new_infections")

    chosen = []
    used_labels = set()
    for idx, label in enumerate(labels):
        if label not in used_labels:
            chosen.append(idx)
            used_labels.add(label)
        if len(chosen) >= MaxExampleRuns:
            break
    while len(chosen) < min(MaxExampleRuns, len(labels)):
        next_idx = len(chosen)
        if next_idx not in chosen:
            chosen.append(next_idx)
        else:
            break

    if not chosen:
        return

    rows = len(chosen)
    fig, axes = plt.subplots(rows, 2, figsize=(10, max(3 * rows, 4)), squeeze=False)
    for row, idx in enumerate(chosen):
        length = int(valid_lengths[idx])
        days = np.arange(length)
        infected = trajectories[idx, :length, infected_idx]
        new_infections = trajectories[idx, :length, new_inf_idx]

        ax1 = axes[row, 0]
        ax2 = axes[row, 1]
        ax1.plot(days, infected)
        ax1.set_title(f"run {idx} | seed={int(seeds[idx])} | {labels[idx]}")
        ax1.set_xlabel("day")
        ax1.set_ylabel("infected")

        ax2.plot(days, new_infections)
        ax2.set_title(f"run {idx} | seed={int(seeds[idx])} | {labels[idx]}")
        ax2.set_xlabel("day")
        ax2.set_ylabel("new infections")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OutputDir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DatasetPath)

    trajectories = np.asarray(dataset["trajectories"], dtype=np.float32)
    valid_lengths = np.asarray(dataset["valid_lengths"], dtype=np.int32)
    observable_names = [str(x) for x in dataset["observable_names"]]
    seeds = np.asarray(dataset["seeds"], dtype=np.int32)
    peak_infected = np.asarray(dataset["peak_infected"], dtype=np.int32)
    peak_day = np.asarray(dataset["peak_day"], dtype=np.int32)
    final_attack_size = np.asarray(dataset["final_attack_size"], dtype=np.int32)
    sim_stopped_early = np.asarray(dataset["sim_stopped_early"], dtype=bool)
    config_json = np.asarray(dataset["config_json"])
    run_paths = [str(x) for x in dataset["run_paths"]]

    n_runs, max_days, n_features = trajectories.shape
    n_agents = infer_n_agents(trajectories, valid_lengths, config_json)
    attack_rates = final_attack_size / n_agents
    labels = attack_rate_labels(attack_rates)

    label_counts = {
        "contained": int(np.sum(labels == "contained")),
        "boundary": int(np.sum(labels == "boundary")),
        "major_outbreak": int(np.sum(labels == "major_outbreak")),
    }

    summary = {
        "dataset_path": str(DatasetPath.resolve()),
        "n_runs": int(n_runs),
        "max_days": int(max_days),
        "n_features": int(n_features),
        "observable_names": observable_names,
        "length_stats": summary_stats(valid_lengths.astype(np.float32)),
        "peak_infected_stats": summary_stats(peak_infected.astype(np.float32)),
        "peak_day_stats": summary_stats(peak_day.astype(np.float32)),
        "attack_rate_stats": summary_stats(attack_rates.astype(np.float32)),
        "label_counts": label_counts,
        "fraction_stopped_early": float(np.mean(sim_stopped_early.astype(np.float32))),
        "seed_min": int(np.min(seeds)) if len(seeds) else None,
        "seed_max": int(np.max(seeds)) if len(seeds) else None,
    }

    (OutputDir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"Dataset: {DatasetPath.resolve()}")
    lines.append(f"Runs: {n_runs}")
    lines.append(f"Trajectory tensor shape: {trajectories.shape}")
    lines.append(f"Observable names: {', '.join(observable_names)}")
    lines.append("")
    lines.append("Trajectory length stats")
    for key, value in summary["length_stats"].items():
        lines.append(f"  {key}: {value:.3f}")
    lines.append("")
    lines.append("Peak infected stats")
    for key, value in summary["peak_infected_stats"].items():
        lines.append(f"  {key}: {value:.3f}")
    lines.append("")
    lines.append("Attack rate stats")
    for key, value in summary["attack_rate_stats"].items():
        lines.append(f"  {key}: {value:.6f}")
    lines.append("")
    lines.append("Regime counts")
    for key, value in label_counts.items():
        lines.append(f"  {key}: {value}")
    lines.append("")
    lines.append("Example runs")
    for idx in range(min(8, n_runs)):
        lines.append(f"  run={idx} seed={int(seeds[idx])} length={int(valid_lengths[idx])} " 
                     f"attack_rate={attack_rates[idx]:.6f} label={labels[idx]} path={run_paths[idx]}")
    write_text_summary(OutputDir / "dataset_summary.txt", "\n".join(lines))

    save_histogram(valid_lengths.astype(np.float32), "Trajectory lengths", "days recorded", OutputDir / "trajectory_lengths.png")
    save_histogram(attack_rates.astype(np.float32), "Final attack rates", "attack rate", OutputDir / "attack_rate_hist.png")
    save_histogram(peak_infected.astype(np.float32), "Peak infected", "peak infected", OutputDir / "peak_infected_hist.png")
    save_example_trajectories(trajectories, valid_lengths, observable_names, labels, seeds, OutputDir / "example_trajectories.png")

    print(f"Inspection written to: {OutputDir.resolve()}")
    print(f"Runs: {n_runs}")
    print(f"Regime counts: {label_counts}")
    print(f"Observable names: {observable_names}")


if __name__ == "__main__":
    main()
