import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from handy.handy_prototype import (
    HANDY_PARAMETER_SYMBOLS,
    HANDY_VARIABLES_SYMBOLS,
    simulate_handy,
)


def sensitivity_analysis(
    parameters_sa_bounds,
    initial_value,
    differential_t,
    simulation_steps,
):
    problem = {
        "num_vars": len(HANDY_PARAMETER_SYMBOLS),
        "names": HANDY_PARAMETER_SYMBOLS,
        "bounds": parameters_sa_bounds,
    }
    sample_parameters = saltelli.sample(problem, 1024, calc_second_order=False)
    simulations = []
    for parameters in sample_parameters:
        try:
            simulations.append(
                simulate_handy(
                    initial_value,
                    parameters,
                    differential_t,
                    simulation_steps,
                    ignore_errors=False,
                ).T
            )
        except:
            simulations.append(simulations[-1])
    simulations = np.array(simulations)

    return {
        symbol: np.array(
            [
                sobol.analyze(
                    problem, simulations[:, symbol_idx, t], calc_second_order=False
                )["S1"]
                for t in range(simulations.shape[-1])
            ]
        )
        for symbol_idx, symbol in enumerate(HANDY_VARIABLES_SYMBOLS)
    }
