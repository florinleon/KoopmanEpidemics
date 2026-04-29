from pathlib import Path
import csv
import math
import matplotlib.pyplot as plt
from config import default_config
from scheduler import run_simulation
from koopman_io import save_run_npz
import pandas as pd


# Choose one of:
#   "single_run"
#   "boundary_dataset"
#   "adaptive_boundary_dataset"
#   "build_dataset"
RunMode = "build_dataset"

# Per-run NPZ files are written here.
RunsDir = Path("runs")

# Aggregated dataset path produced by build_dataset_from_runs().
AggregatedDatasetPath = Path("baseline_dataset.npz")

# Daily end-of-day observables are saved in each NPZ file.
# The 5-day delay window should be built later inside the Koopman module.

# Outcome labeling for boundary studies.
# "major_outbreak" if final attack rate >= AttackRateThreshold, else "contained".
AttackRateThreshold = 0.30

# Settings for a single experiment.
SingleRunSettings = {
    "seed": 50,
    "susceptibility_min": 1.3,
    "susceptibility_max": 1.3019593,
    "write_csv": True,
    "plot": True,
}

# Settings for automated dataset generation near a suspected boundary.
# The idea is to densify runs in the region where outcomes may flip.
BoundaryDatasetSettings = {
    "seeds": list(range(40, 60)),
    "susceptibility_min": 1.3,
    "susceptibility_max_values": [
        1.3019585,
        1.3019588,
        1.3019590,
        1.3019592,
        1.3019593,
        1.3019594,
        1.3019595,
        1.3019597,
        1.3019600,
        1.3019603,
    ],
    "clear_runs_dir_first": False,
    "write_summary_csv": True,
    "plot_examples": False,
}

# Settings for adaptive boundary-focused generation.
# Stage 1: coarse scan over susceptibility_max.
# Stage 2: automatically refine only intervals where the outbreak regime mixes or flips.
AdaptiveBoundarySettings = {
    "seeds": list(range(50, 250)),
    "susceptibility_min": 1.3,
    "coarse_start": 1.3015,  # 1.3019580
    "coarse_stop": 1.3025,  # 1.3019605
    "coarse_num_points": 11,
    "refine_points_per_interval": 4,
    "clear_runs_dir_first": False,
    "write_summary_csv": True,
    "plot_examples": False,
}


# Settings for building one batch NPZ from per-run files.
BuildDatasetSettings = {
    "input_patterns": ["runs/*.npz"],
    "output_path": AggregatedDatasetPath,
}


def make_config():
    cfg = default_config()
    return cfg


def print_summary(metrics):
    days = len(metrics["susceptible"])
    print("Day  Susceptible  Infected  Recovered  Dead")
    for d in range(days):
        print(f"{d:3d}  {metrics['susceptible'][d]:11d}  " 
              f"{metrics['infected'][d]:8d}  {metrics['recovered'][d]:9d}  {metrics['dead'][d]:4d}")


def write_csv(metrics, path):
    keys = list(metrics.keys())
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["day"] + keys)
        for i in range(len(metrics["susceptible"])):
            writer.writerow([i] + [metrics[k][i] for k in keys])


def plot_metrics(metrics, title=None):
    days = list(range(len(metrics["susceptible"])))
    plt.figure()
    plt.plot(days, metrics["susceptible"], label="Susceptible")
    plt.plot(days, metrics["infected"], label="Infected")
    plt.plot(days, metrics["recovered"], label="Recovered")
    plt.plot(days, metrics["dead"], label="Dead")
    plt.xlabel("Day")
    plt.ylabel("Number of Agents")
    if title:
        plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def frange(start, stop, step):
    num_steps = int(round((stop - start) / step))
    for i in range(num_steps + 1):
        yield start + i * step


def linspace_values(start, stop, num_points):
    if num_points < 2:
        return [float(start)]
    step = (stop - start) / (num_points - 1)
    return [float(start + i * step) for i in range(num_points)]


def sweep_susceptibility():
    results = []
    xvals = [round(v, 6) for v in frange(1, 5.5, 0.05)]

    for smax in xvals:
        print(smax)

        cfg = default_config()
        cfg.seed = 9
        cfg.susceptibility_min = 1
        cfg.susceptibility_max = smax
        metrics = run_simulation(cfg)
        final_sus = metrics["susceptible"][-1]
        results.append((smax, final_sus))

    with open("sweep_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["susceptibility_max", "final_susceptible"])
        writer.writerows(results)

    x, y = zip(*results)
    plt.plot(x, y)
    plt.xlabel("Susceptibility Max")
    plt.ylabel("Final Number of Susceptible Agents")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def experiment_quarantine_one_by_one():
    # Try quarantining each agent on each day individually
    cfg = default_config()
    cfg.susceptibility_min = 1.5
    cfg.susceptibility_max = 3.1
    cfg.do_quarantine = True
    total_agents = cfg.n_agents
    total_days = 40

    output_path = "quarantine_results.csv"
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["agent_id", "day", "final_susceptible"])

        for agent_id in range(total_agents):
            for day in range(total_days):
                print(f"Quarantining agent {agent_id} on day {day}")

                def quarantine_fn(current_day, current_agent_id):
                    return current_day == day and current_agent_id == agent_id

                metrics = run_simulation(cfg, quarantine_fn=quarantine_fn)
                final_sus = metrics["susceptible"][-1]
                writer.writerow([agent_id, day, final_sus])
                f.flush()

    # Optional: visualize results for one agent
    df = pd.read_csv(output_path)
    agent_0 = df[df["agent_id"] == 0]
    plt.plot(agent_0["day"], agent_0["final_susceptible"], marker='o')
    plt.xlabel("Quarantine Day (agent 0)")
    plt.ylabel("Final Susceptible Count")
    plt.title("Effect of Quarantining Agent 0 on Different Days")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def outcome_label_from_metrics(metrics, cfg, attack_rate_threshold=AttackRateThreshold):
    n_agents = cfg.n_agents
    final_susceptible = metrics["susceptible"][-1]
    final_attack_rate = (n_agents - final_susceptible) / n_agents
    if final_attack_rate >= attack_rate_threshold:
        return "major_outbreak"
    return "contained"


def ensure_runs_dir(clear_first=False):
    RunsDir.mkdir(parents=True, exist_ok=True)
    if clear_first:
        for path in RunsDir.glob("*.npz"):
            path.unlink()
        for path in RunsDir.glob("*.csv"):
            path.unlink()


def make_run_stem(cfg, run_index=None):
    parts = [
        f"seed_{int(cfg.seed):04d}",
        f"smin_{cfg.susceptibility_min:.7f}",
        f"smax_{cfg.susceptibility_max:.7f}",
    ]
    if run_index is not None:
        parts.insert(0, f"run_{int(run_index):05d}")
    return "__".join(parts).replace(".", "p")


def save_run_outputs(metrics, cfg, *, run_index=None, write_csv_copy=False, run_group=None, stage=None):
    ensure_runs_dir(clear_first=False)
    stem = make_run_stem(cfg, run_index=run_index)
    npz_path = RunsDir / f"{stem}.npz"

    label = outcome_label_from_metrics(metrics, cfg)
    attack_rate = (cfg.n_agents - metrics["susceptible"][-1]) / cfg.n_agents

    extra_fields = {
        "run_label": label,
        "final_attack_rate": attack_rate,
    }
    if run_group is not None:
        extra_fields["run_group"] = run_group
    if stage is not None:
        extra_fields["stage"] = stage

    save_run_npz(npz_path, metrics, cfg, **extra_fields)

    if write_csv_copy:
        csv_path = RunsDir / f"{stem}.csv"
        write_csv(metrics, csv_path)

    return npz_path, label, attack_rate


def run_one_config(cfg, *, run_index, run_group, stage, plot=False):
    print(f"Run {run_index}: seed={cfg.seed}, smin={cfg.susceptibility_min}, " 
          f"smax={cfg.susceptibility_max}, group={run_group}, stage={stage}")
    metrics = run_simulation(cfg)
    npz_path, label, attack_rate = save_run_outputs(metrics, cfg, run_index=run_index, run_group=run_group, stage=stage)

    peak_infected = max(metrics["infected"]) if metrics["infected"] else 0
    peak_day = metrics["infected"].index(peak_infected) if metrics["infected"] else -1
    final_sus = metrics["susceptible"][-1] if metrics["susceptible"] else cfg.n_agents
    n_days = len(metrics["susceptible"])

    row = {
        "run_index": run_index,
        "seed": cfg.seed,
        "susceptibility_min": cfg.susceptibility_min,
        "susceptibility_max": cfg.susceptibility_max,
        "run_group": run_group,
        "stage": stage,
        "npz_path": str(npz_path),
        "outcome_label": label,
        "final_attack_rate": attack_rate,
        "final_susceptible": final_sus,
        "peak_infected": peak_infected,
        "peak_day": peak_day,
        "n_days_recorded": n_days,
    }

    if plot:
        plot_metrics(metrics, title=f"seed={cfg.seed}, smax={cfg.susceptibility_max}")

    return row


def summarize_boundary_by_parameter(summary_rows):
    df = pd.DataFrame(summary_rows)
    if df.empty:
        return df

    grouped = df.groupby("susceptibility_max", as_index=False).agg(n_runs=("outcome_label", "size"), major_fraction=("outcome_label", lambda s: (s == "major_outbreak").mean()), mean_attack_rate=("final_attack_rate", "mean"), min_attack_rate=("final_attack_rate", "min"), max_attack_rate=("final_attack_rate", "max"))
    grouped = grouped.sort_values("susceptibility_max").reset_index(drop=True)
    grouped["mixed_outcomes"] = (grouped["major_fraction"] > 0.0) & (grouped["major_fraction"] < 1.0)
    grouped["boundary_interval_after"] = False

    for idx in range(len(grouped) - 1):
        left = grouped.loc[idx, "major_fraction"]
        right = grouped.loc[idx + 1, "major_fraction"]
        if grouped.loc[idx, "mixed_outcomes"] or grouped.loc[idx + 1, "mixed_outcomes"] or left != right:
            grouped.loc[idx, "boundary_interval_after"] = True

    return grouped


def choose_refinement_values(boundary_summary, refine_points_per_interval):
    refine_values = set()
    if boundary_summary.empty:
        return []

    for idx in range(len(boundary_summary) - 1):
        left = boundary_summary.iloc[idx]
        right = boundary_summary.iloc[idx + 1]
        if not bool(left["boundary_interval_after"]):
            continue

        left_smax = float(left["susceptibility_max"])
        right_smax = float(right["susceptibility_max"])
        if math.isclose(left_smax, right_smax):
            continue

        step = (right_smax - left_smax) / (refine_points_per_interval + 1)
        for k in range(1, refine_points_per_interval + 1):
            refine_values.add(left_smax + k * step)

    return sorted(refine_values)


def write_boundary_summary_files(summary_rows, boundary_summary, prefix):
    summary_path = RunsDir / f"{prefix}_runs_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote run summary to: {summary_path}")

    if boundary_summary is not None and not boundary_summary.empty:
        boundary_path = RunsDir / f"{prefix}_boundary_summary.csv"
        boundary_summary.to_csv(boundary_path, index=False)
        print(f"Wrote boundary summary to: {boundary_path}")


def run_single_experiment():
    settings = SingleRunSettings
    cfg = make_config()
    cfg.seed = settings["seed"]
    cfg.susceptibility_min = settings["susceptibility_min"]
    cfg.susceptibility_max = settings["susceptibility_max"]

    metrics = run_simulation(cfg)

    if settings.get("write_csv", True):
        write_csv(metrics, Path("results.csv"))

    npz_path, label, attack_rate = save_run_outputs(metrics, cfg, run_index=0, write_csv_copy=False)
    print(f"Saved run NPZ to: {npz_path}")
    print(f"Outcome label: {label} | final attack rate: {attack_rate:.3f}")
    print_summary(metrics)

    if settings.get("plot", True):
        plot_metrics(metrics, title=f"seed={cfg.seed}, smax={cfg.susceptibility_max}")


def run_boundary_dataset():
    settings = BoundaryDatasetSettings
    ensure_runs_dir(clear_first=settings.get("clear_runs_dir_first", False))

    seeds = list(settings["seeds"])
    smin = settings["susceptibility_min"]
    smax_values = list(settings["susceptibility_max_values"])
    summary_rows = []
    run_index = 0

    print(f"Writing per-run files into: {RunsDir.resolve()}")

    for seed in seeds:
        for smax in smax_values:
            cfg = make_config()
            cfg.seed = int(seed)
            cfg.susceptibility_min = float(smin)
            cfg.susceptibility_max = float(smax)

            row = run_one_config(cfg, run_index=run_index, run_group="boundary_manual", stage="manual", plot=settings.get("plot_examples", False) and run_index < 3)
            summary_rows.append(row)
            run_index += 1

    if settings.get("write_summary_csv", True):
        boundary_summary = summarize_boundary_by_parameter(summary_rows)
        write_boundary_summary_files(summary_rows, boundary_summary, prefix="manual")

    print(f"Completed {run_index} runs.")


def run_adaptive_boundary_dataset():
    settings = AdaptiveBoundarySettings
    ensure_runs_dir(clear_first=settings.get("clear_runs_dir_first", False))

    seeds = list(settings["seeds"])
    smin = float(settings["susceptibility_min"])
    coarse_values = linspace_values(float(settings["coarse_start"]), float(settings["coarse_stop"]), int(settings["coarse_num_points"]))
    refine_points_per_interval = int(settings["refine_points_per_interval"])

    summary_rows = []
    run_index = 0
    print(f"Writing per-run files into: {RunsDir.resolve()}")
    print("Stage 1: coarse scan")

    for seed in seeds:
        for smax in coarse_values:
            cfg = make_config()
            cfg.seed = int(seed)
            cfg.susceptibility_min = smin
            cfg.susceptibility_max = float(smax)
            row = run_one_config(cfg, run_index=run_index, run_group="boundary_adaptive", stage="coarse", plot=settings.get("plot_examples", False) and run_index < 3)
            summary_rows.append(row)
            run_index += 1

    coarse_boundary_summary = summarize_boundary_by_parameter(summary_rows)
    refine_values = choose_refinement_values(coarse_boundary_summary, refine_points_per_interval)

    if refine_values:
        print("Stage 2: refined scan around mixed or flipping intervals")
        for seed in seeds:
            for smax in refine_values:
                cfg = make_config()
                cfg.seed = int(seed)
                cfg.susceptibility_min = smin
                cfg.susceptibility_max = float(smax)
                row = run_one_config(cfg, run_index=run_index, run_group="boundary_adaptive", stage="refined", plot=False)
                summary_rows.append(row)
                run_index += 1
    else:
        print("No mixed or flipping coarse intervals found. No refinement runs were generated.")

    if settings.get("write_summary_csv", True):
        full_boundary_summary = summarize_boundary_by_parameter(summary_rows)
        write_boundary_summary_files(summary_rows, coarse_boundary_summary, prefix="adaptive_coarse")
        full_summary_path = RunsDir / "adaptive_all_runs_summary.csv"
        pd.DataFrame(summary_rows).to_csv(full_summary_path, index=False)
        print(f"Wrote combined run summary to: {full_summary_path}")
        full_boundary_path = RunsDir / "adaptive_full_boundary_summary.csv"
        full_boundary_summary.to_csv(full_boundary_path, index=False)
        print(f"Wrote refined boundary summary to: {full_boundary_path}")

    print(f"Completed {run_index} runs.")


def build_dataset_from_runs():
    from koopman_io import build_batch_npz

    settings = BuildDatasetSettings
    run_paths = []
    for pattern in settings["input_patterns"]:
        run_paths.extend(sorted(Path().glob(pattern)))

    if not run_paths:
        raise FileNotFoundError("No run NPZ files matched the configured input patterns.")

    output_path = Path(settings["output_path"])
    build_batch_npz(run_paths, output_path)
    print(f"Wrote aggregated dataset to: {output_path.resolve()}")


if __name__ == "__main__":
    if RunMode == "single_run":
        run_single_experiment()
    elif RunMode == "boundary_dataset":
        run_boundary_dataset()
    elif RunMode == "adaptive_boundary_dataset":
        run_adaptive_boundary_dataset()
    elif RunMode == "build_dataset":
        build_dataset_from_runs()
    else:
        raise ValueError(f"Unknown run mode: {RunMode}")
