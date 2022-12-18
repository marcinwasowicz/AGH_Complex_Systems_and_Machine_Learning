import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from handy.handy_prototype import (
    HANDY_PARAMETER_SYMBOLS,
    HANDY_VARIABLES_SYMBOLS,
)


def norm(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def pack_parameters(parameters_json):
    return np.array(
        [float(parameters_json[symbol]) for symbol in HANDY_PARAMETER_SYMBOLS]
    )


def pack_variables(variables_json):
    return np.array(
        [float(variables_json[symbol]) for symbol in HANDY_VARIABLES_SYMBOLS]
    )


def plot_handy(simulation, simulation_steps, differential_t):
    steps_array = [i * differential_t for i in range(simulation_steps)]
    for symbol_idx, symbol in enumerate(HANDY_VARIABLES_SYMBOLS):
        plt.plot(steps_array, norm(simulation.T[symbol_idx]), label=symbol)
    plt.legend()
    plt.show()


def plot_handy_partial(simulation, simulation_steps, differential_t):
    steps_array = [i * differential_t for i in range(simulation_steps)]
    _fig, axs = plt.subplots(2, 2)
    axs[0, 0].plot(steps_array, simulation.T[0])
    axs[0, 0].set_title(HANDY_PARAMETER_SYMBOLS[0])
    axs[0, 1].plot(steps_array, simulation.T[1])
    axs[0, 1].set_title(HANDY_PARAMETER_SYMBOLS[1])
    axs[1, 0].plot(steps_array, simulation.T[2])
    axs[1, 0].set_title(HANDY_PARAMETER_SYMBOLS[2])
    axs[1, 1].plot(steps_array, simulation.T[3])
    axs[1, 1].set_title(HANDY_PARAMETER_SYMBOLS[3])
    plt.show()


def plot_sensitivity_analysis(sensitivity_analysis_result):
    for symbol, symbol_sa in sensitivity_analysis_result.items():
        plt.title(symbol)
        x = [i for i in range(len(symbol_sa))]
        for parameter_symbol_idx, parameter_symbol in enumerate(
            HANDY_PARAMETER_SYMBOLS
        ):
            y = symbol_sa[:, parameter_symbol_idx]
            plt.plot(x, y, label=parameter_symbol)
        plt.legend()
        plt.show()
