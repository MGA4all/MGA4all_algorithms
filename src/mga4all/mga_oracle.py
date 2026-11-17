from math import comb

import numpy as np
import pandas as pd
import pypsa
import cdd

from dataclasses import dataclass
from scipy.optimize import linprog

from .model_interface_pypsa import (
    add_slack_constraint,
    assign_mga_objective,
    create_mga_model,
    extract_diversified_capacity,
    extract_minimum_feasible_cost,
    match_config_techs_to_model_techs,
    convert_linear_weights_into_pypsa,
)
from .utils import (
    alternatives_dict_to_frame,
    save_network,
    normalise_point,
    denormalise_point,
)
from .validate import ORACLEConfig


@dataclass
class OuterApproximation:
    A: np.ndarray
    b: np.ndarray

    lb_physical: pd.Series
    ub_physical: pd.Series

    def contains(self, point, tol=1e-8):
        point = np.asarray(point, dtype=float)
        return np.all(self.A @ point <= self.b + tol)


def setup_mga_model(config: ORACLEConfig, network):
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


def create_target_variables(config: ORACLEConfig, network_mga):
    spatial = False
    target_techs = match_config_techs_to_model_techs(config, network_mga)
    deployed_capacity_series_spatial, deployed_capacity_series_aggregate = (
        extract_diversified_capacity(target_techs, network_mga)
    )
    if spatial:
        deployed_capacity_series = deployed_capacity_series_spatial
    else:
        deployed_capacity_series = deployed_capacity_series_aggregate

    return target_techs, deployed_capacity_series, spatial


def generate_integer_weights(
    deployed_capacity_series: pd.Series, config: ORACLEConfig
) -> pd.DataFrame:
    """Generate targeted weights for each intensified variable using index of deployed_capacity."""
    columns = [
        f"{tech}-{suffix}"
        for tech in config.intensified_technologies
        for suffix in ["max", "min"]
    ]
    weights = pd.DataFrame(
        0.0,
        index=deployed_capacity_series.index,
        columns=columns,
    )

    for tech in weights.index:
        for direction in ["min", "max"]:
            if direction == "min":
                weights.loc[tech, f"{tech}-{direction}"] = 1
            elif direction == "max":
                weights.loc[tech, f"{tech}-{direction}"] = -1
    return weights


def update_mga_objective(
    network_mga, model_mga, mga_weights_series, target_techs, spatial
):
    assign_mga_objective(
        network_mga, model_mga, mga_weights_series, target_techs, spatial=False
    )
    return (network_mga, model_mga)


def initialise_outer_approximation(lb, ub):
    """
    Build H-representation A z <= b of the initial box.
    """

    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    n_dim = len(lb)

    A = []
    b = []

    for i in range(n_dim):

        upper = np.zeros(n_dim)
        upper[i] = 1.0

        lower = np.zeros(n_dim)
        lower[i] = -1.0

        A.append(upper)
        b.append(ub[i])

        A.append(lower)
        b.append(-lb[i])

    return np.asarray(A), np.asarray(b)


def distance_to_inner_hull_linf(vertex, inner_points):
    """
    L-infinity distance from a point to the convex hull
    of inner_points.

    Returns
    -------
    distance : float
    projection : ndarray
    lambdas : ndarray
    """

    vertex = np.asarray(vertex, dtype=float)
    inner_points = np.asarray(inner_points, dtype=float)

    n_points, n_dim = inner_points.shape

    #
    # Variables:
    # [lambda_1 ... lambda_n, t]
    #
    n_vars = n_points + 1
    t_idx = n_points

    c = np.zeros(n_vars)
    c[t_idx] = 1.0

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_points] = 1.0
    b_eq = np.array([1.0])

    A_ub = []
    b_ub = []

    for j in range(n_dim):

        p_j = inner_points[:, j]

        # v_j - y_j <= t
        row = np.zeros(n_vars)
        row[:n_points] = -p_j
        row[t_idx] = -1.0

        A_ub.append(row)
        b_ub.append(-vertex[j])

        # y_j - v_j <= t
        row = np.zeros(n_vars)
        row[:n_points] = p_j
        row[t_idx] = -1.0

        A_ub.append(row)
        b_ub.append(vertex[j])

    bounds = [(0.0, None)] * n_points + [(0.0, None)]

    result = linprog(
        c,
        A_ub=np.asarray(A_ub),
        b_ub=np.asarray(b_ub),
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )

    if not result.success:
        raise RuntimeError(result.message)

    lambdas = result.x[:n_points]
    projection = lambdas @ inner_points
    distance = result.x[t_idx]

    return distance, projection, lambdas


def extract_poi_expressions(
    network_mga,
    model_mga,
    mga_weights,
    target_techs,
):
    """
    Build one Linopy expression per PoI.

    Returns
    -------
    dict[str, linopy.LinearExpression]
        Mapping:
            {
                "gas": expression,
                "solar": expression,
                ...
            }
    """

    pois = sorted({tech for carriers in target_techs.values() for tech in carriers})

    poi_expressions = {}

    for poi in pois:

        # TODO: the current implementation restarts from the
        # weight object created for the min/max initialisation.
        # It may be worth reconstructing the weights from scratch here,
        # so that they are decoupled from the initialisation.
        column = mga_weights.loc[poi].ne(0).idxmax()
        weights = mga_weights[column]

        weight_dict = convert_linear_weights_into_pypsa(
            network=network_mga,
            mga_weights_series=weights,
            target_techs=target_techs,
            spatial=False,
        )

        expr = network_mga.optimize.build_linexpr_from_weights(
            weight_dict,
            model=model_mga,
        )

        poi_expressions[poi] = expr.sum()

    return poi_expressions


def extract_projection_cut_normal(
    model_mga,
    pois,
):
    """
    Recover sigma from the duals of the
    oracle projection constraints.

    Returns
    -------
    pd.Series
    """

    sigma = {}

    for poi in pois:

        upper = model_mga.constraints[f"oracle_upper_{poi}"].dual.item()

        lower = model_mga.constraints[f"oracle_lower_{poi}"].dual.item()

        sigma[poi] = upper - lower

    return pd.Series(sigma, dtype=float)


def enumerate_vertices(A, b):
    """
    Enumerate vertices of

        A x <= b

    using pycddlib.
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # cdd format: b - A x >= 0
    H = np.hstack(
        [
            b.reshape(-1, 1),
            -A,
        ]
    )

    mat = cdd.matrix_from_array(H)
    mat.rep_type = cdd.RepType.INEQUALITY

    poly = cdd.polyhedron_from_matrix(mat)

    generators = cdd.copy_generators(poly)

    vertices = np.array(
        [row[1:] for row in generators.array if row[0] == 1],
        dtype=float,
    )

    return vertices


def normalise_cut(
    sigma,
    rhs,
    lb,
    ub,
):
    """
    Convert a cut

        sigma.T @ z <= rhs

    from physical coordinates into the
    normalized coordinates used by G-ORACLE.

    Parameters
    ----------
    sigma : array-like
    rhs : float
    lb : array-like
    ub : array-like

    Returns
    -------
    sigma_norm : ndarray
    rhs_norm : float
    """

    sigma = np.asarray(sigma, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    ranges = ub - lb
    center = 0.5 * (ub + lb)

    sigma_norm = 0.5 * ranges * sigma

    rhs_norm = rhs - sigma @ center

    return sigma_norm, rhs_norm


# --- Actual Algorithm ---#


def oracle_algorithm(
    config: ORACLEConfig,
    network_costopt: pypsa.Network,
    delta_convergence: float = 0.08,
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

    ## STEP 0. Initialise extreme points via targeted min/max
    mga_weights = generate_integer_weights(deployed_capacity_series, config)
    mga_weights_agnostic = mga_weights.copy()
    mga_weights_agnostic.columns = range(1, len(mga_weights_agnostic.columns) + 1)

    for iteration, weights in mga_weights_agnostic.items():
        mga_weights_series = weights
        network_mga, model_mga = update_mga_objective(
            network_mga, model_mga, mga_weights_series, target_techs, spatially_explicit
        )
        network_mga.optimize.solve_model(log_to_console=False)

        mga_spatial_alternatives[iteration], mga_alternatives[iteration] = (
            extract_diversified_capacity(target_techs, network_mga)
        )

        save_network(export_nc, iteration, network_mga)

    ## STEP 1. Inner and Outer approximations
    points = alternatives_dict_to_frame(mga_alternatives)
    lb = points.min(axis=1)
    ub = points.max(axis=1)
    # Inner points are already available, but we want to use the same convention as
    # the vertices, namely rows=points, columns=dimensions/variables
    inner_points = points.T.to_numpy()
    inner_points_norm = np.array(
        [
            normalise_point(
                p,
                lb.values,
                ub.values,
            )
            for p in inner_points
        ]
    )
    lb_norm = -np.ones(len(lb))
    ub_norm = np.ones(len(lb))
    A, b = initialise_outer_approximation(lb_norm, ub_norm)
    outer = OuterApproximation(
        A=A,
        b=b,
        lb_physical=lb,
        ub_physical=ub,
    )

    delta = delta_convergence
    best_distance = np.inf

    ## STEP 2. Find trial point in Outer Approximation

    iteration = iteration + 1
    while best_distance > delta:
        vertices = enumerate_vertices(
            outer.A,
            outer.b,
        )

        best_distance = -np.inf
        best_vertex = None

        for vertex in vertices:

            distance, projection, lambdas = distance_to_inner_hull_linf(
                vertex,
                inner_points_norm,
            )

            if distance > best_distance:
                best_distance = distance
                best_vertex = vertex

        zO = denormalise_point(
            best_vertex,
            lb.values,
            ub.values,
        )

        zO = pd.Series(
            zO,
            index=points.index,
        )

        ## STEP 3. Find nearest feasible/near-optimal point

        mga_weights_neutral = mga_weights[
            (x for x in mga_weights.columns if "min" in x)
        ]

        mga_weights_neutral.columns = range(1, len(mga_weights_neutral.columns) + 1)
        mga_weights_neutral

        # Recreate MGA model
        network_mga, model_mga = setup_mga_model(config, network_costopt)

        poi_expressions = extract_poi_expressions(
            network_mga, model_mga, mga_weights_neutral, target_techs
        )

        # scalar infinity-norm variable
        t = model_mga.add_variables(
            lower=0,
            name="oracle_projection_distance",
        )

        for poi, expr in poi_expressions.items():

            target = zO[poi]

            # expr - target <= t
            model_mga.add_constraints(
                expr - target <= t,
                name=f"oracle_upper_{poi}",
            )

            # -(expr - target) <= t
            model_mga.add_constraints(
                target - expr <= t,
                name=f"oracle_lower_{poi}",
            )

        # objective: minimise distance to trial point
        model_mga.add_objective(
            t,
            overwrite=True,
        )

        status = network_mga.optimize.solve_model()

        zI = extract_diversified_capacity(
            target_techs,
            network_mga,
        )[1]

        zI_norm = normalise_point(
            zI.values,
            lb.values,
            ub.values,
        )

        t_value = float(model_mga.variables["oracle_projection_distance"].solution)

        ## STEP 4. Update (i.e., cut) the projection

        sigma = extract_projection_cut_normal(
            model_mga,
            pois=mga_weights_neutral.index,
        )

        rhs = sigma @ zI

        sigma_norm, rhs_norm = normalise_cut(
            sigma.values,
            rhs,
            lb.values,
            ub.values,
        )

        # Add the found point to the (normalised) inner points
        inner_points_norm = np.vstack(
            [
                inner_points_norm,
                zI_norm,
            ]
        )

        # If this assertion ever fails, something broke in the projection-cut reconstruction.
        assert sigma @ zO > sigma @ zI

        outer.A = np.vstack(
            [
                outer.A,
                sigma_norm,
            ]
        )

        outer.b = np.append(
            outer.b,
            rhs_norm,
        )

        mga_spatial_alternatives[iteration], mga_alternatives[iteration] = (
            extract_diversified_capacity(target_techs, network_mga)
        )

        save_network(export_nc, iteration, network_mga)
        iteration += 1

    mga_spatial_alternatives = alternatives_dict_to_frame(mga_spatial_alternatives)
    mga_alternatives = alternatives_dict_to_frame(mga_alternatives)

    return mga_spatial_alternatives, mga_alternatives  # , mga_weights
