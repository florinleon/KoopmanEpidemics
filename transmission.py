# Transmission logic for the agent-based disease-spread simulation
# infection.py expects a callable named transmission_outcome(viral_load, susceptibility)
# that returns True if the infection occurs based on viral load and susceptibility

Threshold = 0.5  # infection occurs if dose > Threshold


def transmission_outcome(viral_load, susceptibility, *, threshold=Threshold):
    # if the infection dose exceeds the threshold
    dose = viral_load / 100.0 * susceptibility
    return dose > threshold
