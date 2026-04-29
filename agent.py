from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union
import numpy as np
from config import Config
from loader import get_curve  # provides pre‑computed viral‑load arrays


Cell = Tuple[int, int]  # grid coordinate


@dataclass
class Agent:
    # A single simulated individual

    # --- Immutable fields assigned at construction ---
    cfg: Config
    agent_id: int
    home_id: int
    routine: List[Cell]
    phase: int
    susceptibility: float
    immunity_strength: str

    # --- Dynamic state ---
    pos: Union[Cell, int] = field(init=False)
    infection_day: Optional[int] = field(default=None)
    state: str = field(default="susceptible")
    _viral_curve: Optional[np.ndarray] = field(default=None, repr=False)
    _t: int = field(default=0, repr=False)
    _viral_load: float = field(default=0.0, repr=False)

    rng: np.random.Generator = field(init=False, repr=False)


    def __post_init__(self):
        # Initialize agent location and per-agent RNG
        self.pos = self.home_id
        seed = self.cfg.seed + self.agent_id * 31337
        self.rng = np.random.default_rng(seed)


    def prepare_new_day(self, day):
        # Compute the starting index in the routine for this calendar day
        self.start_idx = (day * self.phase) % self.cfg.routine_length


    def move_to_route_cell(self, step):
        # Move to the grid cell scheduled for the given daytime step
        if not self.is_mobile:
            return
        idx = (self.start_idx + step) % self.cfg.routine_length
        self.pos = self.routine[idx]


    def move_home(self):
        # Place the agent at its home location
        self.pos = self.home_id


    @property
    def viral_load(self):
        return self._viral_load


    @property
    def is_mobile(self):
        # True if the agent is allowed to leave home right now
        if self.state == "dead":
            return False
        if self.state == "infected" and self.viral_load >= self.cfg.homebound_thresh:
            return False
        return True


    def infect(self, current_day):
        # Turn a susceptible agent into an infected one
        if self.state != "susceptible":
            return
        self.state = "infected"
        self.infection_day = current_day
        self._viral_curve = get_curve(self.immunity_strength)
        self._t = 0
        self._viral_load = 0.0


    def advance_viral_load(self):
        # Advance viral load by one simulation step and update health state
        if self.state != "infected" or self._viral_curve is None:
            return

        if self._t < len(self._viral_curve):
            self._viral_load = float(self._viral_curve[self._t])
        else:
            self._viral_load = 0.0

        self._t += 1

        if self._viral_load >= self.cfg.death_thresh:
            self.state = "dead"
            return

        full_day_steps = self.cfg.day_steps + self.cfg.night_steps
        if self._t % full_day_steps == 0 and self._viral_load <= self.cfg.recovery_thresh:
            self.state = "recovered"


    def coin_flip(self, p):
        # Return True with probability p using the per-agent RNG
        return bool(self.rng.random() < p)
