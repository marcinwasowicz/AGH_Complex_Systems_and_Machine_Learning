import numpy as np
from SALib.sample import saltelli
from SALib.analyze import sobol

from handy.handy_prototype import HANDY_PARAMETER_SYMBOLS, simulate_handy


def sensitivity_analysis(
    parameters_sa_bounds,
    initial_value,
    differential_t,
    simulation_steps,
    parameters_sample_count,
):
    problem = {
        "num_vars": len(HANDY_PARAMETER_SYMBOLS),
        "names": HANDY_PARAMETER_SYMBOLS,
        "bounds": parameters_sa_bounds,
    }
    # TODO: Implement sobol SA analysis
    pass
