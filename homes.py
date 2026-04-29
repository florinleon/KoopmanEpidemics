class HomeManager:
    # Manage occupancy of the off-grid homes

    __slots__ = ("n_homes", "_occupants")


    def __init__(self, n_homes):
        self.n_homes = n_homes
        # Map home_id → set of agent_ids currently inside
        self._occupants = {i: set() for i in range(n_homes)}


    def add(self, agent_id, home_id):
        # Place agent_id into home_id (idempotent)
        self._occupants[home_id].add(agent_id)


    def remove(self, agent_id, home_id):
        # Remove agent_id from home_id (no error if absent)
        self._occupants[home_id].discard(agent_id)


    def move(self, agent_id, old_home_id, new_home_id):
        # Move agent_id from old_home_id to new_home_id
        if old_home_id != new_home_id:
            self._occupants[old_home_id].discard(agent_id)
            self._occupants[new_home_id].add(agent_id)


    def occupants(self, home_id):
        # Return a live set of agent IDs currently in home_id
        return self._occupants[home_id]


    def homes_with_agents(self):
        # Iterate over (home_id, occupants_set) pairs that are non-empty
        for hid, occ in self._occupants.items():
            if occ:
                yield hid, occ


    def clear(self):
        # Empty all homes
        for occ in self._occupants.values():
            occ.clear()
