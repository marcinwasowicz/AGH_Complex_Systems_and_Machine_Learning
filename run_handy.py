import json
import sys

sys.path.insert(0, ".")
from handy.handy_prototype import (
    simulate_handy,
)
from utils import pack_parameters, pack_variables, plot_handy


if __name__ == "__main__":
    _script, config_path = sys.argv

    with open(config_path, "r") as config_fd:
        config_json = json.load(config_fd)

    parameters = pack_parameters(config_json["parameters"])
    initial_value = pack_variables(config_json["initial_value"])
    differential_t = float(config_json["differential_t"])
    simulation_steps = config_json["simulation_steps"]

    simulation = simulate_handy(
        initial_value, parameters, differential_t, simulation_steps
    )
    plot_handy(simulation, min(simulation_steps, len(simulation)), differential_t)
