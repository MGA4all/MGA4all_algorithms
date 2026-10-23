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
from .validate import ComboConfig
from .utils import alternatives_dict_to_frame


def setup_mga_model(config: ComboConfig, network):
    minimum_cost, fixed_cost, full_costs = extract_minimum_feasible_cost(network)
    slack = config.cost_slack
    network_mga, model_mga = create_mga_model(network)
    add_slack_constraint(
        model_mga,
        minimum_cost,
        fixed_cost,
        full_costs,
        slack,
        include_non_expandable=True,
    )
    return (network_mga, model_mga)


def create_target_variables(config: ComboConfig, network_mga):
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
    index: pd.Index, alternatives: int
) -> pd.DataFrame:
    """Generate uniformly-random weights using index of deployed_capacity."""
    random_values = np.random.uniform(-1, 1, size=(len(index), alternatives))
    return pd.DataFrame(
        np.round(random_values, 2),
        index=index,
        columns=range(1, alternatives + 1),
    )

def generate_random_integer_weights(
    index: pd.Index, alternatives: int
) -> pd.DataFrame:
    """Generate random weights in the integer set [-1, 0, 1] using index of deployed_capacity."""
    random_values = np.random.choice(
        [-1, 0, 1],
        size=(len(index), alternatives),
    )
    return pd.DataFrame(
        random_values,
        index=index,
        columns=range(1, alternatives + 1),
    )

def generate_combined_weights(
    index: pd.Index,
    config: ComboConfig,
    random_share: float,
    minmax_share: float,
) -> pd.DataFrame:
    """
    Combines random weights generated following a 'random-direction' uniform sampling
    and a 'variable min-max' integer sampling in the set [-1, 0, 1].
    The proportion between the two sampling strategies can be customised.
    By default, it's 75% variable min-max and 25% random-directions.
    """
    if random_share < 0 or minmax_share < 0:
        raise ValueError("Shares must be non-negative.")

    if not np.isclose(random_share + minmax_share, 1.0):
        raise ValueError("random_share and minmax_share must sum to 1.")

    n_random = round(config.alternatives * random_share)
    n_minmax = config.alternatives - n_random

    random_weights = generate_random_weights(index, n_random)
    minmax_weights = generate_random_integer_weights(index, n_minmax)

    combined_weights = pd.concat(
        [random_weights, minmax_weights],
        axis=1,
        ignore_index=True,
    )

    combined_weights = combined_weights.sample(
        axis=1, frac=1
    )

    combined_weights.columns = range(1, config.alternatives + 1)

    return combined_weights


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial
    )
    return (network_mga, model_mga)


def combo_algorithm(
    config: ComboConfig,
    network_costopt: pypsa.Network,
    random_share: float  = 0.25,
    minmax_share: float  = 0.75,
    export_nc: str | bool = False,
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
    mga_weights = generate_combined_weights(
        index=deployed_capacity_series.index, config=config,
        random_share=random_share, minmax_share=minmax_share 
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

        if export_nc == False:
            pass
        elif type(export_nc) == str:
            network_mga.export_to_netcdf(export_nc + "mga_{}.nc".format(iteration))
        else:
            network_mga.export_to_netcdf("mga_{}.nc".format(iteration))

    mga_spatial_alternatives = alternatives_dict_to_frame(mga_spatial_alternatives)
    mga_alternatives = alternatives_dict_to_frame(mga_alternatives)

    return mga_spatial_alternatives, mga_alternatives, mga_weights
