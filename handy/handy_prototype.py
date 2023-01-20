import numpy as np
from scipy.integrate import ode


HANDY_PARAMETER_SYMBOLS = [
    "beta_c",
    "beta_e",
    "alpha_m",
    "alpha_M",
    "s",
    "kappa",
    "rho",
    "delta",
    "lambda_",
    "gamma",
]


HANDY_VARIABLES_SYMBOLS = ["x_c", "x_e", "y", "w"]


def _handy_model_compute_parameters(
    value: np.ndarray,
    num_args: np.ndarray,
    simplify_consumption_rates,
    simplify_death_rates,
):
    """
    value: vector of time-dependent variables in handy model: x_c, x_e, y, w
    args: vector of model hyperparameters:
    beta_c, beta_e, alpha_m, alpha_M, s, kappa, rho, delta, lambda_, gamma
    """
    _beta_c, _beta_e, alpha_m, alpha_M, s, kappa, rho, delta, lambda_, gamma = num_args
    x_c, x_e, _y, w = value

    w_th = rho * x_c + kappa * rho * x_e
    w_coefficient = 1.0 if simplify_consumption_rates else min(1.0, w / w_th)
    C_c = w_coefficient * s * x_c
    C_e = w_coefficient * kappa * s * x_e

    alpha_difference = 0.0 if simplify_death_rates else alpha_M - alpha_m
    alpha_c = alpha_m + max(0.0, 1 - C_c / (s * x_c)) * alpha_difference
    alpha_e = alpha_m + max(0.0, 1 - C_e / (s * x_e)) * alpha_difference

    return alpha_c, alpha_e, C_c, C_e


def _handy_model_ode_f(_t, value: np.ndarray, args):
    num_args, simplification_1, simplification_2 = args
    (
        beta_c,
        beta_e,
        _alpha_m,
        _alpha_M,
        _s,
        _kappa,
        _rho,
        delta,
        lambda_,
        gamma,
    ) = num_args
    x_c, x_e, y, _w = value

    alpha_c, alpha_e, C_c, C_e = _handy_model_compute_parameters(
        value, num_args, simplification_1, simplification_2
    )
    return np.array(
        [
            beta_c * x_c - alpha_c * x_c,  # x_c
            beta_e * x_e - alpha_e * x_e,  # x_e
            gamma * y * (lambda_ - y) - delta * x_c * y,  # y
            delta * x_c * y - C_c - C_e,  # w
        ]
    )


def simulate_handy(
    initial_value: np.ndarray,
    model_parameters: np.ndarray,
    differential_t: float,
    simulation_steps: int,
    ignore_errors=True,
    simplify_consumption_rates=False,
    simplify_death_rates=False,
):
    ode_solver = ode(_handy_model_ode_f).set_integrator("lsoda")
    ode_solver.set_initial_value(initial_value, 0).set_f_params(
        (model_parameters, simplify_consumption_rates, simplify_death_rates)
    )

    simulation = []

    for _ in range(simulation_steps):
        if not ode_solver.successful():
            if ignore_errors:
                break
            raise Exception("Simulation exploded")
        simulation.append(ode_solver.integrate(ode_solver.t + differential_t))
    return np.stack(simulation)
