from dataclasses import dataclass, field
import csv
from pathlib import Path


@dataclass
class MetricsCollector:
    # Accumulate day-by-day counts of epidemiological states

    days = field(default_factory=list)
    susceptible = field(default_factory=list)
    infected = field(default_factory=list)
    recovered = field(default_factory=list)
    dead = field(default_factory=list)


    def record(self, day, agents):
        # Count agent states and append one row of metrics
        s = i = r = d = 0
        for a in agents:
            if a.state == "susceptible":
                s += 1
            elif a.state == "infected":
                i += 1
            elif a.state == "recovered":
                r += 1
            elif a.state == "dead":
                d += 1
        self.days.append(day)
        self.susceptible.append(s)
        self.infected.append(i)
        self.recovered.append(r)
        self.dead.append(d)


    def to_dict(self):
        # Return metrics as a dictionary of lists
        return {
            "day": self.days,
            "susceptible": self.susceptible,
            "infected": self.infected,
            "recovered": self.recovered,
            "dead": self.dead,
        }


    def save_csv(self, path):
        # Write the metrics out to path as CSV with a header row
        with path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["day", "susceptible", "infected", "recovered", "dead"])
            for idx in range(len(self.days)):
                writer.writerow([self.days[idx], self.susceptible[idx], self.infected[idx], self.recovered[idx], self.dead[idx]])


    def to_csv(self, path):
        self.save_csv(path)


def collect_final_summary(mc):
    # Return a concise dictionary with final epidemic statistics
    return {
        "total_infected": sum(mc.infected),
        "peak_infected": max(mc.infected) if mc.infected else 0,
        "total_dead": mc.dead[-1] if mc.dead else 0,
        "total_recovered": mc.recovered[-1] if mc.recovered else 0,
    }
