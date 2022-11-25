import json
import sys

sys.path.insert(0, ".")
from handy.handy_prototype import (
    simulate_handy,
)
from utils import plot_handy, pack_parameters, pack_variables


if __name__ == "__main__":
    simulations = []
    for config in sys.argv[1:]:
        with open(config, "r") as config_fd:
            config_json = json.load(config_fd)

        parameters = pack_parameters(config_json["parameters"])
        initial_value = pack_variables(config_json["initial_value"])
        differential_t = float(config_json["differential_t"])
        simulation_steps = config_json["simulation_steps"]

        simulations.append(
            simulate_handy(initial_value, parameters, differential_t, simulation_steps)
        )
    simulations = [
        simulation[: min([simulation.shape[0] for simulation in simulations]), :]
        for simulation in simulations
    ]
    simulation = sum(simulations)
    plot_handy(simulation, len(simulation), differential_t)
