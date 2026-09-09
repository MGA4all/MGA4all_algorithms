import pandas as pd
import pypsa
import numpy as np

from .model_interface_pypsa import (
    match_config_techs_to_model_techs,
    extract_diversified_capacity,
    extract_minimum_feasible_cost,
    create_mga_model,
    add_slack_constraint,
    assign_mga_objective,
)
from .validate import RandomDirectionsConfig
from .utils import alternatives_dict_to_frame


def setup_mga_model(config: RandomDirectionsConfig, network_costopt):
    network = network_costopt
    minimum_cost = extract_minimum_feasible_cost(network)
    slack = config.cost_slack
    network_mga, model_mga = create_mga_model(network)
    add_slack_constraint(model_mga, minimum_cost, slack)
    return (network_mga, model_mga)


def create_target_variables(config: RandomDirectionsConfig, network_mga):
    spatial = config.spatially_explicit
    target_techs = match_config_techs_to_model_techs(config, network_mga)
    deployed_capacity_series_spatial, deployed_capacity_series_aggregate = (
        extract_diversified_capacity(target_techs, network_mga)
    )
    if spatial == True:
        deployed_capacity_series = deployed_capacity_series_spatial
    else:
        deployed_capacity_series = deployed_capacity_series_aggregate

    return target_techs, deployed_capacity_series, spatial


def generate_random_weights(
    index: pd.Index, config: RandomDirectionsConfig
) -> pd.DataFrame:
    """Generate random weights using index of deployed_capacity."""
    random_values = np.random.uniform(-1, 1, size=(len(index), config.alternatives))
    return pd.DataFrame(
        np.round(random_values, 2),
        index=index,
        columns=range(1, config.alternatives + 1),
    )


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial
    )
    return (network_mga, model_mga)


def random_directions_algorithm(
    config: RandomDirectionsConfig, network_costopt: pypsa.Network
):
    mga_alternatives = {}
    mga_spatial_alternatives = {}

    network_mga, model_mga = setup_mga_model(config, network_costopt)
    target_techs, deployed_capacity_series, spatially_explicit = (
        create_target_variables(config, network_mga)
    )

    mga_spatial_alternatives[0], mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt
    )

    # Pre-compute all randomised weights; helps with parallelisation
    mga_weights = generate_random_weights(
        index=deployed_capacity_series.index, config=config
    )

    for iteration, weights in mga_weights.items():
        mga_weights_series = weights
        network_mga, model_mga = update_mga_objective(
            network_mga, model_mga, mga_weights_series, target_techs, spatially_explicit
        )
        network_mga.optimize.solve_model(log_to_console=False)

        mga_spatial_alternatives[iteration], mga_alternatives[iteration] = (
            extract_diversified_capacity(target_techs, network_mga)
        )

    mga_spatial_alternatives = alternatives_dict_to_frame(mga_spatial_alternatives)
    mga_alternatives = alternatives_dict_to_frame(mga_alternatives)

    return mga_spatial_alternatives, mga_alternatives, mga_weights
