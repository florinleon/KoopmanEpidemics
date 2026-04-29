from collections import defaultdict
from random import Random
import csv
from config import Config, default_config
from grid import Grid
from homes import HomeManager
from infection import process_all
from agent import Agent


__all__ = ["run_simulation"]

EnableLogging = False
LogPath = "contact_log.csv"


def _setup(cfg, rng):
    # Create objects that the simulation needs
    grid = Grid(cfg.grid_size)
    homes = HomeManager(cfg.n_homes)
    agents = _load_agents(cfg, rng)
    rng.choice(agents).infect(current_day=0)
    metrics = defaultdict(list)
    return grid, homes, agents, metrics


def _load_agents(cfg, rng):
    # Load agents using the loader factory
    from loader import load_agents
    return load_agents(cfg, rng)


def _open_contact_logger():
    if not EnableLogging:
        return None, None
    fh = open(LogPath, "w", newline="")
    writer = csv.writer(fh)
    writer.writerow(["day", "segment", "step", "location_type", "x_or_home", "y", "agents"])
    return fh, writer


def _log_multi_occupancy(day, segment, step, grid, homes, writer):
    if writer is None:
        return
    for (x, y), occ in grid.occupied_cells():
        if len(occ) > 1:
            writer.writerow([day, segment, step, "grid", x, y, ";".join(map(str, occ))])
    for home_id, occ in homes.homes_with_agents():
        if len(occ) > 1:
            writer.writerow([day, segment, step, "home", home_id, "", ";".join(map(str, occ))])


def _count_end_of_day_states(agents):
    # Return susceptible, infected, recovered, dead, infected_mobile, infected_homebound
    s = i = r = d = 0
    infected_mobile = infected_homebound = 0

    for ag in agents:
        match ag.state:
            case "susceptible":
                s += 1
            case "infected":
                i += 1
                if ag.is_mobile:
                    infected_mobile += 1
                else:
                    infected_homebound += 1
            case "recovered":
                r += 1
            case _:
                d += 1

    return s, i, r, d, infected_mobile, infected_homebound


def run_simulation(cfg=None, *, seed=None, quarantine_fn=None, day_start_recorder=None, day_end_recorder=None, return_agents=False):
    # Run cfg.sim_days days and return the collected metrics.
    #
    # Optional hooks are intentionally passive: they receive the current
    # in-memory agents but must not mutate them unless the caller explicitly
    # wants to alter the simulation. They make it possible to build supervised
    # features for intervention models without changing the simulator dynamics.
    if cfg is None:
        cfg = default_config()

    rng = Random(seed if seed is not None else cfg.seed)
    grid, homes, agents, metrics = _setup(cfg, rng)
    log_fh, log_writer = _open_contact_logger()

    day_steps, night_steps = cfg.day_steps, cfg.night_steps
    prev_recovered = 0
    prev_dead = 0

    for day in range(cfg.sim_days):
        new_inf_today = 0

        for ag in agents:
            ag.prepare_new_day(day)

            if cfg.do_quarantine:
                ag._quarantine_today = quarantine_fn(day, ag.agent_id) if quarantine_fn else False
            else:
                ag._quarantine_today = False

        if day_start_recorder is not None:
            day_start_recorder(day, agents)

        for step in range(day_steps):
            for ag in agents:
                if ag.state == "dead":
                    continue

                if isinstance(ag.pos, tuple):
                    grid.remove(ag.agent_id, ag.pos)
                else:
                    homes.remove(ag.agent_id, ag.pos)

                if ag._quarantine_today:
                    ag.move_home()
                else:
                    ag.move_to_route_cell(step)

                if isinstance(ag.pos, tuple):
                    grid.add(ag.agent_id, ag.pos)
                else:
                    homes.add(ag.agent_id, ag.pos)

            new_inf_today += process_all(grid, homes, agents, rng, day)
            _log_multi_occupancy(day, "day", step, grid, homes, log_writer)
            for ag in agents:
                ag.advance_viral_load()

        for step in range(night_steps):
            for ag in agents:
                if ag.state == "dead":
                    continue

                if isinstance(ag.pos, tuple):
                    grid.remove(ag.agent_id, ag.pos)
                else:
                    homes.remove(ag.agent_id, ag.pos)

                ag.move_home()
                homes.add(ag.agent_id, ag.pos)

            new_inf_today += process_all(grid, homes, agents, rng, day)
            _log_multi_occupancy(day, "night", step, grid, homes, log_writer)
            for ag in agents:
                ag.advance_viral_load()

        s, i, r, d, infected_mobile, infected_homebound = _count_end_of_day_states(agents)
        new_recoveries = r - prev_recovered
        new_deaths = d - prev_dead
        prev_recovered = r
        prev_dead = d

        metrics["susceptible"].append(s)
        metrics["infected"].append(i)
        metrics["recovered"].append(r)
        metrics["dead"].append(d)
        metrics["new_infections"].append(new_inf_today)
        metrics["new_recoveries"].append(new_recoveries)
        metrics["new_deaths"].append(new_deaths)
        metrics["infected_mobile"].append(infected_mobile)
        metrics["infected_homebound"].append(infected_homebound)

        if day_end_recorder is not None:
            day_end_recorder(day, agents, metrics)

        if i == 0:
            print(f"No infected agents remaining after day {day}. Stopping early.")
            break

    if log_fh:
        log_fh.close()
        print(f"Contact log written to {LogPath}")

    if return_agents:
        return metrics, agents
    return metrics
