from pathlib import Path
import random
import numpy as np
from config import Config, default_config


_curve_cache = {}  # strength: np.ndarray


def get_curve(strength, cfg=None):
    # Return the viral-load array for the given strength (in percent)
    if cfg is None:
        cfg = default_config()

    fname = cfg.viral_load_files[strength]
    if strength not in _curve_cache:
        data = np.genfromtxt(Path(fname), delimiter=",", usecols=1, dtype=float)
        _curve_cache[strength] = data
    return _curve_cache[strength]


def load_agents(cfg, rng):
    # Instantiate cfg.n_agents and return them as a list
    from agent import Agent  # local import to avoid circular dependency

    agents = []
    home_ids = list(range(cfg.n_homes))
    agent_homes = [rng.choice(home_ids) for _ in range(cfg.n_agents)]

    strengths = ["strong", "medium", "low", "compromised"]
    probs = [0.35, 0.45, 0.15, 0.05]

    for agent_id in range(cfg.n_agents):
        home_id = agent_homes[agent_id]
        susc = rng.uniform(cfg.susceptibility_min, cfg.susceptibility_max)
        immune = rng.choices(strengths, probs, k=1)[0]

        routine, seen = [], set()
        while len(routine) < cfg.routine_length:
            cell = (rng.randrange(cfg.grid_size), rng.randrange(cfg.grid_size))
            if cell in seen:
                continue
            seen.add(cell)
            routine.append(cell)

        phase = rng.randrange(1, cfg.routine_length)

        agents.append(Agent(cfg, agent_id, home_id, routine, phase, susc, immune))

    return agents
