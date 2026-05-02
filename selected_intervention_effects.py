import contextlib
import copy
import csv
import io
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt


SelectedInterventionsCsv = "selected_interventions.csv"
OutputDir = "selected_intervention_effects"
SusceptibilityMin = 1.302
SusceptibilityMax = 1.303
SimDays = 365
OutbreakThreshold = 0.3
ImageDpi = 180
SuppressSimulatorOutput = True


# The program is meant to sit next to the simulation files, but this search also supports the common case where the simulator was extracted into a nearby folder. This keeps the script free of command-line arguments while still making it practical to run after downloading the project archive.
def LocateProjectRoot():
    ScriptDirectory = Path(__file__).resolve().parent
    CurrentDirectory = Path.cwd().resolve()
    CandidateDirectories = [ScriptDirectory, CurrentDirectory, ScriptDirectory / "Simulation", ScriptDirectory / "Simulation_src", CurrentDirectory / "Simulation", CurrentDirectory / "Simulation_src"]
    for CandidateDirectory in CandidateDirectories:
        if (CandidateDirectory / "scheduler.py").exists() and (CandidateDirectory / "config.py").exists():
            return CandidateDirectory.resolve()
    raise FileNotFoundError("Could not find scheduler.py and config.py. Place this program in the simulator directory, or next to a Simulation/Simulation_src directory.")


# The intervention CSV is intentionally located independently from the code root. In practice, users often keep selected_interventions.csv next to this reporting script while the simulator modules live in a project subdirectory.
def LocateSelectedInterventionsCsv(ProjectRoot):
    ScriptDirectory = Path(__file__).resolve().parent
    CurrentDirectory = Path.cwd().resolve()
    CandidateFiles = [ScriptDirectory / SelectedInterventionsCsv, CurrentDirectory / SelectedInterventionsCsv, ProjectRoot / SelectedInterventionsCsv, ProjectRoot / "threshold_seed_outbreak_search" / SelectedInterventionsCsv, ScriptDirectory / "threshold_seed_outbreak_search" / SelectedInterventionsCsv, CurrentDirectory / "threshold_seed_outbreak_search" / SelectedInterventionsCsv]
    for CandidateFile in CandidateFiles:
        if CandidateFile.exists():
            return CandidateFile.resolve()
    raise FileNotFoundError("Could not find selected_interventions.csv. Place it next to this program, in the simulator directory, or in threshold_seed_outbreak_search/.")


# The simulator loads viral-load CSV files through the Config object. Absolute paths prevent accidental failure when this program is launched from a directory other than the simulator source directory.
def MakeConfig(ProjectRoot, seed, susceptibility_min, susceptibility_max, sim_days, do_quarantine):
    from config import default_config
    BaseConfig = default_config()

    class RuntimeConfig:
        pass

    ConfigObject = RuntimeConfig()
    for Name in dir(BaseConfig):
        if Name.startswith("_"):
            continue
        Value = getattr(BaseConfig, Name)
        if callable(Value):
            continue
        setattr(ConfigObject, Name, copy.deepcopy(Value))
    ConfigObject.seed = int(seed)
    ConfigObject.susceptibility_min = float(susceptibility_min)
    ConfigObject.susceptibility_max = float(susceptibility_max)
    ConfigObject.sim_days = int(sim_days)
    ConfigObject.do_quarantine = bool(do_quarantine)
    ConfigObject.viral_load_files = {Strength: str((ProjectRoot / FileName).resolve()) if not Path(FileName).is_absolute() else str(Path(FileName).resolve()) for Strength, FileName in ConfigObject.viral_load_files.items()}
    return ConfigObject


# Each selected row names a seed plus one agent-day intervention. Empty columns from spreadsheet exports are discarded so that a trailing unnamed column cannot alter the simulation inputs.
def ReadSelectedInterventions(CsvPath):
    Rows = []
    with CsvPath.open("r", newline="", encoding="utf-8-sig") as FileHandle:
        Reader = csv.DictReader(FileHandle)
        for RawRow in Reader:
            Row = {}
            for Key, Value in RawRow.items():
                if Key is None:
                    continue
                CleanKey = str(Key).strip()
                if not CleanKey:
                    continue
                Row[CleanKey] = str(Value).strip() if Value is not None else ""
            if Row:
                Rows.append(Row)
    return Rows


# The baseline and counterfactual must be rerun from the same seed and parameterization. This preserves the random homes, routines, susceptibility values, immunity classes, and index case, so the plotted difference is attributable to the one-day mobility intervention rather than to a fresh stochastic draw.
def RunMetrics(ProjectRoot, seed, susceptibility_min, susceptibility_max, sim_days, agent_id, intervention_day, do_quarantine):
    from scheduler import run_simulation
    ConfigObject = MakeConfig(ProjectRoot, seed, susceptibility_min, susceptibility_max, sim_days, do_quarantine)

    def QuarantineFunction(current_day, current_agent_id):
        return int(current_day) == int(intervention_day) and int(current_agent_id) == int(agent_id)

    OriginalDirectory = Path.cwd()
    os.chdir(ProjectRoot)
    try:
        if SuppressSimulatorOutput:
            with contextlib.redirect_stdout(io.StringIO()):
                if do_quarantine:
                    return run_simulation(ConfigObject, quarantine_fn=QuarantineFunction)
                return run_simulation(ConfigObject)
        if do_quarantine:
            return run_simulation(ConfigObject, quarantine_fn=QuarantineFunction)
        return run_simulation(ConfigObject)
    finally:
        os.chdir(OriginalDirectory)


# Final attack rate and peak infection burden are the two most useful summary values for this visual diagnostic. The attack rate determines whether the case remains above the outbreak threshold, while the peak indicates how much instantaneous epidemic burden was avoided.
def SummarizeMetrics(Metrics, n_agents):
    Susceptible = list(Metrics.get("susceptible", []))
    Infected = list(Metrics.get("infected", []))
    if not Susceptible:
        return {"final_attack_rate": 0.0, "final_susceptible": int(n_agents), "peak_infected": 0, "peak_day": -1}
    PeakInfected = max(Infected) if Infected else 0
    PeakDay = Infected.index(PeakInfected) if Infected else -1
    FinalSusceptible = int(Susceptible[-1])
    return {"final_attack_rate": float((int(n_agents) - FinalSusceptible) / float(n_agents)), "final_susceptible": FinalSusceptible, "peak_infected": int(PeakInfected), "peak_day": int(PeakDay)}


# Intervention trajectories often terminate earlier because the infection chain has been eliminated. For the visual comparison, the counterfactual series is padded with zeros so both curves share the same day axis and the extinction result remains visible through the end of the baseline run.
def PadInterventionSeries(InterventionValues, TargetLength):
    PaddedValues = list(InterventionValues)
    if len(PaddedValues) < int(TargetLength):
        PaddedValues.extend([0] * (int(TargetLength) - len(PaddedValues)))
    return PaddedValues


# This figure focuses on active infections only. The cumulative attack-rate plot was intentionally removed so that each selected case is represented by a single panel comparing the baseline outbreak trajectory with the post-intervention trajectory.
def PlotCase(OutputPath, CaseIndex, Seed, AgentId, InterventionDay, BaselineMetrics, InterventionMetrics, BaselineSummary, InterventionSummary):
    BaselineInfected = list(BaselineMetrics.get("infected", []))
    InterventionInfected = PadInterventionSeries(InterventionMetrics.get("infected", []), len(BaselineInfected))
    BaselineDays = list(range(len(BaselineInfected)))
    InterventionDays = list(range(len(InterventionInfected)))
    PeakReduction = max(BaselineSummary["peak_infected"] - InterventionSummary["peak_infected"], 0)
    Figure, Axis = plt.subplots(1, 1, figsize=(11.5, 5.4))
    Figure.suptitle(f"Case {int(CaseIndex):02d}, seed {int(Seed)}: quarantine agent {int(AgentId)} on day {int(InterventionDay)}", fontsize=13, fontweight="bold")
    Axis.plot(BaselineDays, BaselineInfected, label="Baseline active infected", linewidth=2.0)
    Axis.plot(InterventionDays, InterventionInfected, label="After intervention active infected", linewidth=2.0)
    Axis.axvline(int(InterventionDay), linestyle=":", linewidth=1.5, label="Intervention day")
    Axis.set_xlabel("Simulation day")
    Axis.set_ylabel("Active infected agents")
    Axis.grid(True, alpha=0.3)
    Axis.legend(loc="best")
    SummaryText = f"Baseline: peak {BaselineSummary['peak_infected']} on day {BaselineSummary['peak_day']}, final attack rate {BaselineSummary['final_attack_rate']:.3f}    |    Intervention: peak {InterventionSummary['peak_infected']} on day {InterventionSummary['peak_day']}, final attack rate {InterventionSummary['final_attack_rate']:.3f}    |    Peak reduction: {PeakReduction} agents"
    Figure.text(0.5, 0.025, SummaryText, ha="center", va="bottom", fontsize=9)
    Figure.tight_layout(rect=[0.03, 0.08, 0.99, 0.93])
    OutputPath.parent.mkdir(parents=True, exist_ok=True)
    Figure.savefig(OutputPath, dpi=ImageDpi)
    plt.close(Figure)


# A selected row may come from a minimal CSV that has no susceptibility columns. The constants above reproduce the near-threshold sweep used for the three uploaded cases, while optional CSV columns still override them if future selections include explicit values.
def ValueFromRow(Row, Name, DefaultValue, CastFunction):
    Value = Row.get(Name, "")
    if Value == "":
        return CastFunction(DefaultValue)
    return CastFunction(Value)


# This function deliberately creates one image per selected case and no aggregate report file. The separate figures make each intervention effect easy to inspect, attach to a manuscript folder, or regenerate without post-processing.
def BuildAllFigures():
    ProjectRoot = LocateProjectRoot()
    if str(ProjectRoot) not in sys.path:
        sys.path.insert(0, str(ProjectRoot))
    CsvPath = LocateSelectedInterventionsCsv(ProjectRoot)
    OutputDirectory = (Path(__file__).resolve().parent / OutputDir).resolve()
    Rows = ReadSelectedInterventions(CsvPath)
    from config import default_config
    NumberOfAgents = int(default_config().n_agents)
    for Row in Rows:
        CaseIndex = ValueFromRow(Row, "case_index", len(Row), int)
        Seed = ValueFromRow(Row, "seed", 0, int)
        AgentId = ValueFromRow(Row, "best_agent_id", 0, int)
        InterventionDay = ValueFromRow(Row, "best_day", 0, int)
        RowSusceptibilityMin = ValueFromRow(Row, "susceptibility_min", SusceptibilityMin, float)
        RowSusceptibilityMax = ValueFromRow(Row, "susceptibility_max", SusceptibilityMax, float)
        RowSimDays = ValueFromRow(Row, "sim_days", SimDays, int)
        BaselineMetrics = RunMetrics(ProjectRoot, Seed, RowSusceptibilityMin, RowSusceptibilityMax, RowSimDays, AgentId, InterventionDay, False)
        InterventionMetrics = RunMetrics(ProjectRoot, Seed, RowSusceptibilityMin, RowSusceptibilityMax, RowSimDays, AgentId, InterventionDay, True)
        BaselineSummary = SummarizeMetrics(BaselineMetrics, NumberOfAgents)
        InterventionSummary = SummarizeMetrics(InterventionMetrics, NumberOfAgents)
        OutputPath = OutputDirectory / f"intervention_effect_case_{int(CaseIndex):02d}_seed_{int(Seed)}_agent_{int(AgentId)}_day_{int(InterventionDay)}.png"
        PlotCase(OutputPath, CaseIndex, Seed, AgentId, InterventionDay, BaselineMetrics, InterventionMetrics, BaselineSummary, InterventionSummary)


if __name__ == "__main__":
    BuildAllFigures()
