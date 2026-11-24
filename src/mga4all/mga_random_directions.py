import numpy as np
import pandas as pd
import pypsa

from .direction_similarity_checks import (
    is_different_enough,
    perturb_noise,
    perturb_rank_flip,
)
from .model_interface_pypsa import (
    add_slack_constraint,
    assign_mga_objective,
    create_mga_model,
    extract_diversified_capacity,
    extract_minimum_feasible_cost,
    match_config_techs_to_model_techs,
    solve_network,
)
from .utils import alternatives_dict_to_frame, save_network
from .validate import RandomDirectionsConfig


def setup_mga_model(config: RandomDirectionsConfig, network):
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


def create_target_variables(config: RandomDirectionsConfig, network_mga):
    spatial = config.spatially_explicit
    target_techs = match_config_techs_to_model_techs(config, network_mga)
    deployed_capacity_series_spatial, deployed_capacity_series_aggregate = (
        extract_diversified_capacity(target_techs, network_mga)
    )
    if spatial:
        deployed_capacity_series = deployed_capacity_series_spatial
    else:
        deployed_capacity_series = deployed_capacity_series_aggregate

    return target_techs, deployed_capacity_series, spatial


def generate_random_weights(
    index: pd.Index,
    config: RandomDirectionsConfig,
    adaptive: bool = False,
) -> pd.DataFrame:
    """Generate random weights using index of deployed_capacity."""
    random_values = np.random.uniform(-1, 1, size=(len(index), config.alternatives))
    candidate_weights = pd.DataFrame(
        np.round(random_values, 2),
        index=index,
        columns=range(1, config.alternatives + 1),
    )

    MAX_NOISE_ATTEMPTS = 50

    if adaptive:
        accepted = {}
        accepted[1] = candidate_weights[1]

        for col in candidate_weights.columns[1:]:
            weights = candidate_weights[col]

            if is_different_enough(
                weights,
                accepted,
                cosine_threshold=0.97,
                ranking_threshold=0.9,
            ):
                accepted[col] = weights
            else:
                # 1. Ranking perturbation first
                candidate = perturb_rank_flip(weights)
                # 2. Noise perturbation if needed
                attempt = 0
                while (
                    not is_different_enough(candidate, accepted)
                    and attempt < MAX_NOISE_ATTEMPTS
                ):
                    candidate = perturb_noise(
                        candidate, noise_scale=0.03 * (attempt + 1)
                    )
                    attempt += 1
                # Final accepted candidate
                accepted[col] = candidate
        candidate_weights = alternatives_dict_to_frame(accepted)
    else:
        pass

    return candidate_weights


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial
    )
    return (network_mga, model_mga)


def random_directions_algorithm(
    config: RandomDirectionsConfig,
    network_costopt: pypsa.Network,
    adaptive: bool = False,
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
    mga_weights = generate_random_weights(
        index=deployed_capacity_series.index, config=config, adaptive=adaptive
    )

    for iteration, weights in mga_weights.items():
        mga_weights_series = weights
        network_mga, model_mga = update_mga_objective(
            network_mga, model_mga, mga_weights_series, target_techs, spatially_explicit
        )
        solve_network(network_mga)

        mga_spatial_alternatives[iteration], mga_alternatives[iteration] = (
            extract_diversified_capacity(target_techs, network_mga)
        )

        save_network(export_nc, iteration, network_mga)

    mga_spatial_alternatives = alternatives_dict_to_frame(mga_spatial_alternatives)
    mga_alternatives = alternatives_dict_to_frame(mga_alternatives)

    return mga_spatial_alternatives, mga_alternatives, mga_weights
