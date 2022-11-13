import json
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from handy.handy_prototype import (
    simulate_handy,
    HANDY_PARAMETER_SYMBOLS,
    HANDY_VARIABLES_SYMBOLS,
)


def norm(arr):
    return (arr - np.min(arr)) / (np.min(arr) - np.max(arr))


def pack_parameters(parameters_json):
    return np.array(
        [float(parameters_json[symbol]) for symbol in HANDY_PARAMETER_SYMBOLS]
    )


def pack_variables(variables_json):
    return np.array(
        [float(variables_json[symbol]) for symbol in HANDY_VARIABLES_SYMBOLS]
    )


def plot_handy(simulation, simulation_steps):
    steps_array = [i * differential_t for i in range(simulation_steps)]
    for symbol_idx, symbol in enumerate(HANDY_VARIABLES_SYMBOLS):
        plt.plot(steps_array, norm(simulation.T[symbol_idx]), label=symbol)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    _script, config_path = sys.argv

    with open(config_path, "r") as config_fd:
        config_json = json.load(config_fd)

    parameters = pack_parameters(config_json["parameters"])
    initial_value = pack_variables(config_json["initial_value"])
    differential_t = float(config_json["differential_t"])
    simulation_steps = config_json["simulation_steps"]
    parameters_sa_bounds = config_json["parameters_sa_bounds"]
    sa_parameters_sample_count = config_json["sa_parameters_sample_count"]

    simulation = simulate_handy(
        initial_value, parameters, differential_t, simulation_steps
    )
    plot_handy(simulation, min(simulation_steps, len(simulation)))
