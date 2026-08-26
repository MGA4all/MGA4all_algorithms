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
    deployed_capacity_series = extract_diversified_capacity(
        target_techs, network_mga, spatial
    )
    return target_techs, deployed_capacity_series, spatial


def generate_random_weights(index: pd.Index, config: RandomDirectionsConfig) -> pd.DataFrame:
    """Generate random weights using index of deployed_capacity."""
    random_values = np.random.uniform(-1, 1, size=(config.alternatives, len(index))).T
    return pd.DataFrame(random_values, index=index, columns=range(1, config.alternatives+1))

def compute_random_weights(config, deployed_capacity_series):
    mga_weights = generate_random_weights(index=deployed_capacity_series.index, config=config).round(2)
    return mga_weights


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

    mga_spatial_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=True
    )
    mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=False
    )

    # Pre-compute all randomised weights; helps with parallelisation
    mga_weights = compute_random_weights(config, deployed_capacity_series)
    
    for iteration, weights in mga_weights.items():
        mga_weights_series = weights
        network_mga, model_mga = update_mga_objective(
            network_mga, model_mga, mga_weights_series, target_techs, spatially_explicit
        )
        network_mga.optimize.solve_model(log_to_console=False)

        # Storing capacity results for further inspection
        ## TODO: make the result saving and inspection smoother and
        ## standardised across methods
        mga_alternatives[iteration] = extract_diversified_capacity(
            target_techs, network_mga, spatial=False
        )
        mga_spatial_alternatives[iteration] = extract_diversified_capacity(
            target_techs, network_mga, spatial=True
        )

    return mga_alternatives, mga_spatial_alternatives, mga_weights.to_dict()
