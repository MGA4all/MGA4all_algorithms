import pandas as pd
import pypsa
import numpy as np
from pandas.api.types import is_number
from scipy.stats import spearmanr

from .model_interface_pypsa import (
    match_config_techs_to_model_techs,
    extract_diversified_capacity,
    extract_intensified_capacity,
    extract_minimum_feasible_cost,
    create_mga_model,
    add_slack_constraint,
    assign_mga_objective,
)
from .validate import SPORESConfig

from .direction_similarity_checks import (
    ranking_similarity,
    angular_similarity,
    perturb_rank_flip,
    perturb_noise,
    is_different_enough,
)


def setup_mga_model(config: SPORESConfig, network):
    minimum_cost = extract_minimum_feasible_cost(network)
    slack = config.cost_slack
    network_mga, model_mga = create_mga_model(network)
    add_slack_constraint(model_mga, minimum_cost, slack)
    return network_mga, model_mga


def create_target_variables(config: SPORESConfig, network_mga):
    spatial = config.spatially_explicit
    target_techs = match_config_techs_to_model_techs(config, network_mga)
    diversified_technologies_series = extract_diversified_capacity(
        target_techs, network_mga, spatial
    )
    return target_techs, diversified_technologies_series, spatial


def create_intensification_variables(network_mga, spatial, target_techs, config):
    intensified_technologies_series = extract_intensified_capacity(
        target_techs, config, network_mga, spatial
    )
    return intensified_technologies_series


def normalise_l2(weights_series):
    weights_series = weights_series / np.sqrt((weights_series**2).sum())
    return weights_series


def compute_diversification_weights(
    deployed_capacity_series,
    previous_weights_series,
):
    new_weights_series = deployed_capacity_series
    new_weights_series[:] = normalise_l2(deployed_capacity_series).round(2)
    updated_weights_series = previous_weights_series + new_weights_series
    # Perturb if all values are the same
    diversification_weights_series = normalise_l2(updated_weights_series)
    return diversification_weights_series


def compute_intensification_weights(intensified_technologies_series):
    if not is_number(intensified_technologies_series):
        intensification_weights_series = intensified_technologies_series
    else:
        intensification_weights_series = 0
    return intensification_weights_series


# FIXME: move into validation
def compute_coefficients(config: SPORESConfig):
    if isinstance(config.intensification_coefficient, int):
        intensify_coeff = abs(config.intensification_coefficient)
    else:
        intensify_coeff = 1

    return intensify_coeff, config.diversification_coefficient


def compute_combined_weights(
    intensification_weights_series,
    diversification_weights_series,
    intensify_coeff,
    diversify_coeff,
):
    if isinstance(intensification_weights_series, int):
        combined_weights_series = normalise_l2(
            (diversify_coeff * diversification_weights_series)
            + (intensify_coeff * intensification_weights_series)
        )
    else:
        combined_weights_series = normalise_l2(
            pd.concat(
                (
                    (diversify_coeff * diversification_weights_series),
                    (intensify_coeff * intensification_weights_series),
                ),
                axis=1,
            ).sum(axis=1)
        )
    # Check if direction is good, else perturb
    return combined_weights_series


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial
    )
    return (network_mga, model_mga)


def spores_algorithm(
    config: SPORESConfig, network_costopt: pypsa.Network, adaptive: bool = False
):

    MAX_NOISE_ATTEMPTS = 50

    mga_alternatives = {}
    mga_spatial_alternatives = {}
    mga_weights = {}

    network_mga, model_mga = setup_mga_model(config, network_costopt)
    target_techs, diversified_technologies_series, spatially_explicit = (
        create_target_variables(config, network_mga)
    )
    intensified_technologies_series = create_intensification_variables(
        network_mga, spatially_explicit, target_techs, config
    )
    intensify_coeff, diversify_coeff = compute_coefficients(config)

    mga_weights[0] = pd.Series(0, index=diversified_technologies_series.index)
    mga_diversification_weights = pd.Series(
        0, index=diversified_technologies_series.index
    )
    mga_spatial_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=True
    )
    mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt, spatial=False
    )

    intensification_weights_series = compute_intensification_weights(
        intensified_technologies_series
    )

    for iteration in range(1, config.alternatives + 1):

        # If intensification is required, the first iteration
        # implements only pure intensification to find the extreme
        if (
            iteration == 1
            and config.intensification_coefficient != 0
            and isinstance(intensification_weights_series, pd.Series)
        ):
            diversification_weights_series = pd.Series(
                0, index=diversified_technologies_series.index
            )
            mga_weights_series = compute_combined_weights(
                intensification_weights_series,
                diversification_weights_series,
                intensify_coeff,
                diversify_coeff,
            )

        # If intensification is not required, simple diversification applies
        elif iteration == 1 and (
            config.intensification_coefficient == 0
            or intensification_weights_series == 0
        ):
            previous_weights_series = mga_diversification_weights
            if spatially_explicit:
                diversified_technologies_series = mga_spatial_alternatives[
                    iteration - 1
                ].copy()
            else:
                diversified_technologies_series = mga_alternatives[iteration - 1].copy()

            diversification_weights_series = compute_diversification_weights(
                diversified_technologies_series,
                previous_weights_series,
            )
            mga_weights_series = diversification_weights_series

        # In all other cases, both intensification and diversification are
        # accounted for, and the coefficients determine their importance
        else:
            previous_weights_series = mga_diversification_weights
            if spatially_explicit:
                diversified_technologies_series = mga_spatial_alternatives[
                    iteration - 1
                ].copy()
            else:
                diversified_technologies_series = mga_alternatives[iteration - 1].copy()

            diversification_weights_series = compute_diversification_weights(
                diversified_technologies_series,
                previous_weights_series,
            )
            mga_weights_series = compute_combined_weights(
                intensification_weights_series,
                diversification_weights_series,
                intensify_coeff,
                diversify_coeff,
            )

            # ---SMART part----#
            if adaptive:
                # Check if different enough and perturb if not
                if not is_different_enough(mga_weights_series, mga_weights):
                    # 1. Ranking perturbation first
                    candidate = perturb_rank_flip(mga_weights_series)
                    # 2. Noise perturbation if needed
                    attempt = 0
                    while (
                        not is_different_enough(candidate, mga_weights)
                        and attempt < MAX_NOISE_ATTEMPTS
                    ):
                        candidate = perturb_noise(
                            candidate, noise_scale=0.03 * (attempt + 1)
                        )
                        attempt += 1
                    # Final accepted candidate
                    mga_weights_series = candidate
            else:
                pass
            # ---End of SMART part---#

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
        mga_weights[iteration] = mga_weights_series.copy()
        mga_diversification_weights = diversification_weights_series

    return mga_alternatives, mga_spatial_alternatives, mga_weights
