class Grid:
    # A mutable occupancy map for agents moving on a 2-D lattice

    __slots__ = ("_size", "_cells")


    def __init__(self, size):
        self._size = size
        # Only store cells that contain at least one agent to keep memory low
        self._cells = {}


    @property
    def size(self):
        # Return the width / height of the grid
        return self._size


    def _in_bounds(self, xy):
        x, y = xy
        return 0 <= x < self._size and 0 <= y < self._size


    def add(self, agent_id, xy):
        # Place agent_id in cell xy, ignoring out-of-bounds cells
        if not self._in_bounds(xy):
            return
        self._cells.setdefault(xy, set()).add(agent_id)


    def remove(self, agent_id, xy):
        # Remove agent_id from cell xy if present
        cell = self._cells.get(xy)
        if cell is None:
            return
        cell.discard(agent_id)
        if not cell:
            del self._cells[xy]


    def move(self, agent_id, old_xy, new_xy):
        # Move agent_id from old_xy to new_xy in a single operation
        if old_xy == new_xy and self._in_bounds(old_xy):
            return
        self.remove(agent_id, old_xy)
        self.add(agent_id, new_xy)


    def occupants(self, xy):
        # Yield IDs of agents currently in xy
        return iter(self._cells.get(xy, ()))


    def occupied_cells(self):
        # Yield pairs ((x, y), set_of_agents) for all non-empty cells
        for xy, agents in self._cells.items():
            yield xy, agents


    def clear(self):
        # Empty the entire grid
        self._cells.clear()


    def population(self):
        # Return the total number of agents currently on the grid
        return sum(len(agents) for agents in self._cells.values())


    def snapshot(self):
        # Return a copy of the occupancy map {(x, y): [agent_ids, …]}
        return {xy: list(agents) for xy, agents in self._cells.items()}
