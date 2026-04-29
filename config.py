class Config:
    # Parameter set for the multi-agent disease-spread simulation.

    # --- Geometry and population size ---
    grid_size = 50          # width and height of the square grid
    n_homes = 200           # number of off-grid home cells
    n_agents = 500          # size of the agent population

    # --- Daily movement schedule ---
    routine_length = 10     # cells visited in an agent’s daily route
    day_steps = 10          # moves during daytime
    night_steps = 10        # moves at night (agent stays home)

    # --- Simulation horizon ---
    sim_days = 365          # total number of calendar days to simulate
    
    # --- Quarantine simulation ---
    do_quarantine = False

    # --- Disease thresholds and biology ---
    # lower/upper bounds of susceptibility
    susceptibility_min = 1.5    
    susceptibility_max = 3   

    symptom_thresh = 10         # ≥10 % viral load → symptomatic
    homebound_thresh = 50       # ≥50 % viral load → stay home
    death_thresh = 100          # ≥100 % viral load → death
    recovery_thresh = 1         # ≤1 % viral load → potential recovery at day’s end

    # --- Viral-load curve CSVs ---
    viral_load_files = {
        "strong": "viral_load_1_high.csv",
        "medium": "viral_load_2_medium.csv",
        "low": "viral_load_3_low.csv",
        "compromised": "viral_load_4_none.csv",
    }

    # --- Reproducibility ---
    # single global RNG seed
    seed = 42                   



def default_config():
    # Return a Config-like object using static class-level defaults
    return Config
