import json
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from handy.handy_prototype import (
    simulate_handy,
    HANDY_PARAMETER_SYMBOLS,
    HANDY_VARIABLES_SYMBOLS,
)
from handy.handy_sa_analysis import sensitivity_analysis


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


def plot_sensitivity_analysis(sensitivity_analysis_result):
    with open("expensive_calc_results/sensitivity_analysis", "wb+") as sa_fd:
        pickle.dump(sensitivity_analysis_result, sa_fd)

    for symbol, symbol_sa in enumerate(sensitivity_analysis_result.items()):
        plt.title(symbol)
        x = [i for i in range(len(symbol_sa))]
        for parameter_symbol_idx, parameter_symbol in enumerate(
            HANDY_PARAMETER_SYMBOLS
        ):
            y = symbol_sa[:, parameter_symbol_idx]
            plt.plot(x, y, label=parameter_symbol)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    _script, config_path, command = sys.argv

    assert command in [
        "SA",
        "RUN",
    ], f"Unrecognized command, SA for sensitivity analysis, and RUN for plotting simulation"

    with open(config_path, "r") as config_fd:
        config_json = json.load(config_fd)

    parameters = pack_parameters(config_json["parameters"])
    initial_value = pack_variables(config_json["initial_value"])
    differential_t = float(config_json["differential_t"])
    simulation_steps = config_json["simulation_steps"]

    if command == "RUN":
        simulation = simulate_handy(
            initial_value, parameters, differential_t, simulation_steps
        )
        plot_handy(simulation, min(simulation_steps, len(simulation)))
    if command == "SA":
        parameters_sa_bounds = config_json["parameters_sa_bounds"]
        sa = sensitivity_analysis(
            parameters_sa_bounds, initial_value, differential_t, simulation_steps
        )
        plot_sensitivity_analysis(sa)
