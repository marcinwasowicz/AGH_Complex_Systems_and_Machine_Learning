import json
import sys

from auto_esn.esn.esn import GroupedDeepESN, DeepESN
import numpy as np
import torch

sys.path.insert(0, ".")
from handy.handy_prototype import (
    simulate_handy,
)
from utils import pack_parameters, pack_variables, plot_handy_partial


def prepare_training_data(simulation, window_start, window_end, train_fraction):
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
    transform, inverse_transform = prepare_transformations(simulation)

    for config in [(4500, 5500, 0.9)]:
        window_start, window_end, train_fraction = config
        X_train, y_train, test, initial_extrapolation_state = prepare_training_data(
            simulation, window_start, window_end, train_fraction
        )

        model = DeepESN(
            input_size=4,
            num_layers=3,
            hidden_size=1000,
        ).float()

        model.fit(
            torch.from_numpy(transform(X_train)).float(),
            torch.from_numpy(transform(y_train)).float(),
        )

        extrapolation = evaluate_model(
            model,
            len(test),
            torch.from_numpy(transform(initial_extrapolation_state)).float(),
        )
        plot_handy_partial(transform(test), len(test), differential_t)

        plot_handy_partial(
            extrapolation,
            len(extrapolation),
            differential_t,
        )
