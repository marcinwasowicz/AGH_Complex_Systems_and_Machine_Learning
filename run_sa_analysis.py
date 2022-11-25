import json
import pickle
import sys

sys.path.insert(0, ".")
from handy.handy_sa_analysis import sensitivity_analysis
from utils import pack_variables, pack_parameters, plot_sensitivity_analysis


if __name__ == "__main__":
    _script, config_path = sys.argv

    with open(config_path, "r") as config_fd:
        config_json = json.load(config_fd)

    parameters = pack_parameters(config_json["parameters"])
    initial_value = pack_variables(config_json["initial_value"])
    differential_t = float(config_json["differential_t"])
    simulation_steps = config_json["simulation_steps"]
    parameters_sa_bounds = config_json["parameters_sa_bounds"]

    try:
        with open("expensive_calc_results/sensitivity_analysis", "rb") as sa_fd:
            sa = pickle.load(sa_fd)
    except Exception as e:
        print(e)
        sa = sensitivity_analysis(
            parameters_sa_bounds, initial_value, differential_t, simulation_steps
        )
        with open("expensive_calc_results/sensitivity_analysis", "wb+") as sa_fd:
            pickle.dump(sa, sa_fd)

    plot_sensitivity_analysis(sa)
