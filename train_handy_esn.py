import json
import sys

from auto_esn.esn.esn import GroupedDeepESN
import numpy as np
import torch

sys.path.insert(0, ".")
from handy.handy_prototype import (
    simulate_handy,
)
from utils import pack_parameters, pack_variables, plot_handy


def split_data(simulation, window_start, window_end, train_fraction):
    data = simulation[window_start:window_end]

    split_idx = int((window_end - window_start) * train_fraction)
    train = data[:split_idx]
    test = data[split_idx:]

    X_train = train[0 : len(train) - 1, :]
    y_train = train[1 : len(train), :]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    initial_extrapolation_state = train[-1:, :]

    return X_train, y_train, test, initial_extrapolation_state


def prepare_transformations(simulation):
    MIN_ARR = np.min(simulation, axis=0)
    MAX_ARR = np.max(simulation, axis=0)
    return (
        lambda x: (x - MIN_ARR) / (MAX_ARR - MIN_ARR),
        lambda x: x * (MAX_ARR - MIN_ARR) + MIN_ARR,
    )


def evaluate_model(model, extrapolation_steps, state):
    result = []
    for _ in range(extrapolation_steps):
        state = model(state)
        result.append(state)
    return torch.vstack(result).detach().numpy()


def train_and_evaluate_model(x, y, test, transform, inverse_transform, state, repeats):
    best_extrapolation = None
    best_mae = np.inf

    for _ in range(repeats):
        model = GroupedDeepESN(
            groups=3,
            input_size=4,
            num_layers=(3, 3, 3),
            hidden_size=500,
        ).float()

        model.fit(
            torch.from_numpy(transform(x)).float(),
            torch.from_numpy(transform(y)).float(),
        )
        extrapolation = inverse_transform(
            evaluate_model(
                model,
                len(test),
                torch.from_numpy(transform(state)).float(),
            )
        )

        mae = np.mean(np.abs(extrapolation - test))
        if mae < best_mae:
            best_mae = mae
            best_extrapolation = extrapolation

    return best_extrapolation


if __name__ == "__main__":
    _script, config_path = sys.argv

    with open(config_path, "r") as config_fd:
        config_json = json.load(config_fd)

    parameters = pack_parameters(config_json["parameters"])
    initial_value = pack_variables(config_json["initial_value"])
    differential_t = float(config_json["differential_t"])
    simulation_steps = config_json["simulation_steps"]

    baseline_simulation = simulate_handy(
        initial_value, parameters, differential_t, simulation_steps
    )
    simulation_length = len(baseline_simulation)
    transform, inverse_transform = prepare_transformations(baseline_simulation)

    surogate_simulation_1 = simulate_handy(
        initial_value,
        parameters,
        differential_t,
        simulation_steps,
        simplify_consumption_rates=True,
        simplify_death_rates=False,
    )
    surogate_simulation_2 = simulate_handy(
        initial_value,
        parameters,
        differential_t,
        simulation_steps,
        simplify_consumption_rates=True,
        simplify_death_rates=True,
    )
    surogate_simulation_3 = simulate_handy(
        initial_value,
        parameters,
        differential_t,
        simulation_steps,
        simplify_consumption_rates=False,
        simplify_death_rates=True,
    )

    for config in [
        (
            int(0.45 * simulation_length),
            int(0.56 * simulation_length),
            0.91,
            "100_next",
        ),
        (int(0.45 * simulation_length), simulation_length, 0.18, "till_end"),
    ]:
        window_start, window_end, train_fraction, name = config

        _, _, surogate_result_1, _ = split_data(
            surogate_simulation_1, window_start, window_end, train_fraction
        )
        _, _, surogate_result_2, _ = split_data(
            surogate_simulation_2, window_start, window_end, train_fraction
        )
        _, _, surogate_result_3, _ = split_data(
            surogate_simulation_3, window_start, window_end, train_fraction
        )

        X_train, y_train, test, initial_extrapolation_state = split_data(
            baseline_simulation, window_start, window_end, train_fraction
        )

        esn_extrapolation = train_and_evaluate_model(
            x=X_train,
            y=y_train,
            test=test,
            transform=transform,
            inverse_transform=inverse_transform,
            state=initial_extrapolation_state,
            repeats=20,
        )

        plot_handy(test, len(test), differential_t, path=f"./plots/baseline_{name}")
        plot_handy(
            esn_extrapolation,
            len(esn_extrapolation),
            differential_t,
            path=f"./plots/esn_{name}",
        )
        plot_handy(
            surogate_result_1,
            len(surogate_result_1),
            differential_t,
            path=f"./plots/sur_1_{name}",
        )
        plot_handy(
            surogate_result_2,
            len(surogate_result_2),
            differential_t,
            path=f"./plots/sur_2_{name}",
        )
        plot_handy(
            surogate_result_3,
            len(surogate_result_3),
            differential_t,
            path=f"./plots/sur_3_{name}",
        )

        with open(f"./plots/mae_{name}.txt", "w+") as mae_f:
            mae_f.write(
                f"MAE of ESN extrapolation: {np.mean(np.abs(test.T - esn_extrapolation.T), axis=-1)}\n"
            )
            mae_f.write(
                f"MAE of surogate 1 extrapolation: {np.mean(np.abs(test.T - surogate_result_1.T), axis=-1)}\n"
            )
            mae_f.write(
                f"MAE of surogate 2 extrapolation: {np.mean(np.abs(test.T - surogate_result_2.T), axis=-1)}\n"
            )
            mae_f.write(
                f"MAE of surogate 3 extrapolation: {np.mean(np.abs(test.T - surogate_result_3.T), axis=-1)}\n"
            )
