import numpy as np
from agent import Agent
from grid import Grid
from homes import HomeManager
from transmission import transmission_outcome


def process_all(grid, homes, agents, rng, current_day):
    # Evaluate every location, apply the infection rule,
    # and return the number of newly infected agents in this step
    id2agent = {a.agent_id: a for a in agents}
    newly_infected = 0

    # Grid cells
    for _, occ in grid.occupied_cells():
        newly_infected += _handle_group(occ, id2agent, rng, current_day)

    # Homes (uncomment if infection in homes is desired)
    # for _, occ in homes.homes_with_agents():
    #     newly_infected += _handle_group(occ, id2agent, rng, current_day)

    return newly_infected


def _handle_group(occ, id2agent, rng, day):
    # Try all (infected, susceptible) pairs in the location and
    # apply the transmission rule. Return number of new infections
    if len(occ) < 2:
        return 0

    newly = 0
    occ_ids = list(occ)

    for src_id in occ_ids:
        src = id2agent[src_id]
        if src.state != "infected":
            continue
        vload = src.viral_load
        for tgt_id in occ_ids:
            if tgt_id == src_id:
                continue
            tgt = id2agent[tgt_id]
            if tgt.state != "susceptible":
                continue
            if transmission_outcome(vload, tgt.susceptibility):
                tgt.infect(current_day=day)
                newly += 1

    return newly
