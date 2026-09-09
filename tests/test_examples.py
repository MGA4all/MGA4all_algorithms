import yaml

import pytest

from mga4all.examples import create_pypsa_network
from mga4all.mga_hop_skip_jump import hop_skip_jump_algorithm
from mga4all.mga_random_directions import random_directions_algorithm
from mga4all.mga_spores import spores_algorithm
from mga4all.validate import HopSkipJumpConfig, RandomDirectionsConfig, SPORESConfig


@pytest.fixture(scope="module")
def network():
    """Fixture for repeated use of the example pypsa network."""
    network = create_pypsa_network()
    network.optimize()
    return network


@pytest.fixture(scope="module")
def hop_skip_jump_config():
    """Fixture for basic hop-skip-jump configuration."""
    with open("configs/test_config_hop_skip_jump.yaml") as f:
        return HopSkipJumpConfig.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def random_directions_config():
    """Fixture for basic random-directions configuration."""
    with open("configs/test_config_random_directions.yaml") as f:
        return RandomDirectionsConfig.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def spores_diversify_config():
    """Fixture for SPORES configuration using only diversification."""
    with open("configs/test_config_spores_diversify_only.yaml") as f:
        return SPORESConfig.model_validate(yaml.safe_load(f))


@pytest.fixture(scope="module")
def spores_intensify_config():
    """Fixture for SPORES configuration using diversification and intensification."""
    with open("configs/test_config_spores_diversify_intensify.yaml") as f:
        return SPORESConfig.model_validate(yaml.safe_load(f))


@pytest.mark.parametrize(
    ["configuration", "algorithm", "adaptive"],
    [
        pytest.param(
            "hop_skip_jump_config", hop_skip_jump_algorithm, None, id="hop-skip-jump"
        ),
        pytest.param(
            "random_directions_config",
            random_directions_algorithm,
            None,
            id="random-directions",
        ),
        pytest.param(
            "spores_diversify_config", spores_algorithm, False, id="spores-diversify"
        ),
        pytest.param(
            "spores_diversify_config",
            spores_algorithm,
            True,
            id="spores-diversify-adaptive",
        ),
        pytest.param(
            "spores_intensify_config", spores_algorithm, False, id="spores-intensify"
        ),
        pytest.param(
            "spores_intensify_config",
            spores_algorithm,
            True,
            id="spores-intensify-adaptive",
        ),
    ],
)
def test_examples(network, configuration, algorithm, adaptive, request):
    config = request.getfixturevalue(configuration)

    if adaptive is None:
        algorithm(config, network)
    else:
        algorithm(config, network, adaptive=adaptive)
