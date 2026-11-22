from dataclasses import dataclass
from fractions import Fraction

import cdd
import cdd.gmp as cdd_gmp
import numpy as np
import pandas as pd
import pypsa
from scipy.optimize import linprog
from scipy.spatial import cKDTree

from .model_interface_pypsa import (
    add_slack_constraint,
    assign_mga_objective,
    build_oracle_projection_problem,
    create_mga_model,
    extract_diversified_capacity,
    extract_minimum_feasible_cost,
    extract_oracle_poi_expressions,
    extract_oracle_projection_cut_normal,
    match_config_techs_to_model_techs,
    solve_network,
)
from .utils import (
    alternatives_dict_to_frame,
    denormalise_point,
    normalise_point,
    save_network,
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
    spatial = config.spatially_explicit
    target_techs = match_config_techs_to_model_techs(config, network_mga)
    deployed_capacity_series_spatial, deployed_capacity_series_aggregate = (
        extract_diversified_capacity(target_techs, network_mga)
    )
    if spatial:
        deployed_capacity_series = deployed_capacity_series_spatial
    else:
        deployed_capacity_series = deployed_capacity_series_aggregate

    return (
        target_techs,
        deployed_capacity_series,
        deployed_capacity_series_aggregate,
        spatial,
    )


def generate_integer_weights(
    deployed_capacity_series: pd.Series, config: ORACLEConfig
) -> pd.DataFrame:
    """Generate targeted weights for each intensified variable using index of deployed_capacity."""
    columns = [
        f"{tech}-{suffix}"
        for tech in deployed_capacity_series.index
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
        network_mga, model_mga, mga_weights_series, target_techs, spatial
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
    n_vars = n_points + 1
    t_idx = n_points

    c = np.zeros(n_vars)
    c[t_idx] = 1.0

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n_points] = 1.0
    b_eq = np.array([1.0])

    A_ub, b_ub = [], []

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

    # Reconstruct the dual vector associated with the projection.
    #
    # The inequalities were added in pairs:
    #   row 0: v_j - y_j <= t
    #   row 1: y_j - v_j <= t
    #
    # SciPy returns their dual marginals in the same order.
    marginals = result.ineqlin.marginals

    pi = -marginals[0::2] + marginals[1::2]

    return distance, projection, lambdas, pi


# Initialise once, at module level
# enumerate_vertices_gmp_fallback_count = 0


def enumerate_vertices(A, b):
    #    global enumerate_vertices_gmp_fallback_count

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)

    # cdd inequality convention: b - A @ x >= 0
    H = np.column_stack([b, -A])

    # First try floating-point arithmetic.
    try:
        mat = cdd.matrix_from_array(H)
        mat.rep_type = cdd.RepType.INEQUALITY

#        try:
#            cdd.matrix_canonicalize(mat)
#        except RuntimeError:
#            pass

        poly = cdd.polyhedron_from_matrix(mat)
        generators = cdd.copy_generators(poly)

        vertices = np.array(
            [row[1:] for row in generators.array if row[0] == 1],
            dtype=float,
        ).reshape(-1, A.shape[1])

        if len(vertices) == 0:
            raise RuntimeError("Vertex enumeration returned no vertices.")
        if len(vertices):
            violations = A @ vertices.T - b[:, None]
            max_violation = np.max(violations)

            if max_violation > 1e-7:
                raise RuntimeError(
                    "Floating-point vertex validation failed. "
                    f"Maximum violation: {max_violation:.3e}"
                )

        return vertices

    except RuntimeError as exc:
        if not any(
            message in str(exc)
            for message in [
                "Numerical inconsistency",
                "returned no vertices",
                "Floating-point vertex validation failed",
                "LP cycling",
                "Criss-Cross",
            ]
        ):
            raise

    #        enumerate_vertices_gmp_fallback_count += 1

    # GMP fallback
    H_fraction = [[Fraction(str(value)) for value in row] for row in H]

    mat = cdd_gmp.matrix_from_array(
        H_fraction,
        rep_type=cdd_gmp.RepType.INEQUALITY,
    )

    poly = cdd_gmp.polyhedron_from_matrix(mat)
    generators = cdd_gmp.copy_generators(poly)

    vertices = np.array(
        [
            [float(value) for value in row[1:]]
            for row in generators.array
            if row[0] == 1
        ],
        dtype=float,
    ).reshape(-1, A.shape[1])

    if len(vertices):
        violations = A @ vertices.T - b[:, None]
        max_violation = np.max(violations)

        if max_violation > 1e-7:
            raise RuntimeError(
                "GMP vertex validation failed. "
                f"Maximum violation: {max_violation:.3e}"
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
    max_iterations: int = 200,
    export_nc: str | bool = False,
):

    spatial = config.spatially_explicit
    mga_alternatives = {}
    mga_spatial_alternatives = {}

    network_mga, model_mga = setup_mga_model(config, network_costopt)
    (
        target_techs,
        deployed_capacity_series,
        deployed_capacity_series_aggregate,
        spatially_explicit,
    ) = create_target_variables(config, network_mga)

    mga_spatial_alternatives[0], mga_alternatives[0] = extract_diversified_capacity(
        target_techs, network_costopt
    )

    # ------------------------------------------------------------
    # STEP 0. Initialise extreme points via targeted min/max
    # ------------------------------------------------------------

    if spatial:
        mga_weights = generate_integer_weights(deployed_capacity_series, config)
    else:
        mga_weights = generate_integer_weights(
            deployed_capacity_series_aggregate, config
        )

    mga_weights_agnostic = mga_weights.copy()
    mga_weights_agnostic.columns = range(1, len(mga_weights_agnostic.columns) + 1)

    for iteration, weights in mga_weights_agnostic.items():

        network_mga, model_mga = update_mga_objective(
            network_mga, model_mga, weights, target_techs, spatial
        )

        solve_network(network_mga)

        mga_spatial_alternatives[iteration], mga_alternatives[iteration] = (
            extract_diversified_capacity(target_techs, network_mga)
        )

        save_network(export_nc, iteration, network_mga)

    # ------------------------------------------------------------
    # STEP 1. Initialise inner and outer approximations
    # ------------------------------------------------------------

    if spatial:
        points = alternatives_dict_to_frame(mga_spatial_alternatives)
    else:
        points = alternatives_dict_to_frame(mga_alternatives)

    lb = points.min(axis=1)
    ub = points.max(axis=1)
    ranges = ub - lb

    # Check that normalisation is well-defined.
    if not np.isfinite(ranges).all():
        raise ValueError("Non-finite variable ranges in initial alternatives.")

    zero_range = ranges[ranges == 0]

    if not zero_range.empty:
        raise ValueError(
            "Cannot normalise variables with zero range. "
            f"Constant variables: {zero_range.index.tolist()}"
        )

    # Warn about very small ranges, which can cause numerical instability.
    small_ranges = ranges[ranges < 1e-8]

    if not small_ranges.empty:
        print("WARNING: Very small variable ranges:")
        print(small_ranges)

    # Rows = points, columns = variables
    inner_points = points.T.to_numpy()

    inner_points_norm = np.array(
        [normalise_point(p, lb.values, ub.values) for p in inner_points]
    )

    if not np.isfinite(inner_points_norm).all():
        raise ValueError("Non-finite values in normalised initial alternatives.")

    # Initial outer approximation: [-1, 1]^n
    lb_norm = -np.ones(len(lb))
    ub_norm = np.ones(len(lb))

    A, b = initialise_outer_approximation(lb_norm, ub_norm)

    outer = OuterApproximation(
        A=A,
        b=b,
        lb_physical=lb,
        ub_physical=ub,
    )

    best_distance = np.inf

    # ------------------------------------------------------------
    # STEP 2. Find trial point in outer approximation
    # ------------------------------------------------------------

    iteration += 1

    metrics = []
    previous_vertex_cache = []

    # The most recently added inner point.
    # It is None during the first iteration.
    pending_inner_point = None

    # Matching tolerance for vertices across iterations.
    vertex_match_tol = 1e-9

    # Numerical tolerance for the supporting-hyperplane test.
    support_tol = 1e-10

    while best_distance > delta_convergence and iteration <= max_iterations:

        print(f"\nIteration {iteration}")

        # Record the current geometry before adding another cut/point.
        n_outer_constraints = len(outer.b)
        n_inner_points = len(inner_points_norm)

        # --------------------------------------------------------
        # Vertex enumeration
        # --------------------------------------------------------

        vertices = enumerate_vertices(outer.A, outer.b)
        n_vertices = len(vertices)

        print("Number of outer constraints:", n_outer_constraints)
        print("Number of outer vertices:", n_vertices)

        # --------------------------------------------------------
        # Distance calculations with supporting-hyperplane reuse
        # --------------------------------------------------------

        best_distance = -np.inf
        best_vertex = None

        current_vertex_cache = []

        n_cached_distances = 0
        n_distance_lp_solves = 0

        # Build a spatial index of the vertices from the previous iteration.
        if previous_vertex_cache:

            old_vertices = np.array(
                [entry["vertex"] for entry in previous_vertex_cache]
            )

            old_vertex_tree = cKDTree(old_vertices)

        else:
            old_vertex_tree = None

        for vertex in vertices:

            cached_entry = None

            # ----------------------------------------------------
            # 1. Find whether this vertex existed previously
            # ----------------------------------------------------

            if old_vertex_tree is not None:

                match_distance, match_idx = old_vertex_tree.query(vertex)

                if match_distance <= vertex_match_tol:

                    old_entry = previous_vertex_cache[match_idx]

                    # ------------------------------------------------
                    # 2. Check whether the newly added inner point
                    #    invalidates the previous projection
                    # ------------------------------------------------

                    if pending_inner_point is not None:

                        lhs = old_entry["pi"] @ pending_inner_point
                        rhs = old_entry["pi"] @ old_entry["projection"]

                        # Use a conservative test: if the condition is
                        # close to equality, solve the LP again.
                        if lhs <= rhs - support_tol:
                            cached_entry = old_entry

            # ----------------------------------------------------
            # 3. Reuse the old result or solve the distance LP
            # ----------------------------------------------------

            if cached_entry is not None:

                distance = cached_entry["distance"]
                projection = cached_entry["projection"]
                pi = cached_entry["pi"]

                n_cached_distances += 1

            else:

                distance, projection, lambdas, pi = distance_to_inner_hull_linf(
                    vertex,
                    inner_points_norm,
                )

                if np.linalg.norm(pi, ord=1) > 1 + 1e-6:
                    raise RuntimeError("Unexpected distance LP dual vector.")

                n_distance_lp_solves += 1

            # Store the result for the next iteration.
            current_vertex_cache.append(
                {
                    "vertex": vertex.copy(),
                    "distance": distance,
                    "projection": projection.copy(),
                    "pi": pi.copy(),
                    "reused": cached_entry is not None,
                }
            )

            # Update the farthest vertex.
            if distance > best_distance:

                best_distance = distance
                best_vertex = vertex.copy()

        # --------------------------------------------------------
        # DIAGNOSTIC: Validate randomly selected cached distances
        # --------------------------------------------------------

        n_validate = min(20, n_cached_distances)

        if n_validate > 0:

            cached_indices = [
                i for i, entry in enumerate(current_vertex_cache) if entry["reused"]
            ]

            rng = np.random.default_rng(iteration)

            selected = rng.choice(
                cached_indices,
                size=n_validate,
                replace=False,
            )

            errors = []

            for idx in selected:

                entry = current_vertex_cache[idx]

                fresh_distance, _, _, _ = distance_to_inner_hull_linf(
                    entry["vertex"],
                    inner_points_norm,
                )

                errors.append(abs(fresh_distance - entry["distance"]))

            print(
                "Maximum cached-distance validation error:",
                max(errors),
            )

        assert best_vertex is not None
        print("Best distance:", best_distance)

        zO = denormalise_point(
            best_vertex,
            lb.values,
            ub.values,
        )

        zO = pd.Series(zO, index=points.index)

        # --------------------------------------------------------
        # STEP 3. Find nearest feasible / near-optimal point
        # --------------------------------------------------------

        if not spatial:

            mga_weights_neutral = mga_weights[
                [x for x in mga_weights.columns if "min" in x]
            ].copy()

            mga_weights_neutral.columns = range(1, len(mga_weights_neutral.columns) + 1)

        else:

            mga_weights_neutral = pd.DataFrame(
                0.0,
                index=deployed_capacity_series.index,
                columns=range(len(deployed_capacity_series.index)),
            )

            for j, poi in enumerate(mga_weights_neutral.index):
                mga_weights_neutral.loc[poi, j] = 1.0

        # Recreate MGA model
        network_mga, model_mga = setup_mga_model(config, network_costopt)

        poi_expressions = extract_oracle_poi_expressions(
            network_mga,
            model_mga,
            mga_weights_neutral,
            target_techs,
            spatial,
        )

        # FIXME: Operations on the model should be part of the model
        # interface and thus model agnostics. Move 'add_variables' and
        # add_constraints to the model interface.

        # Scalar infinity-norm variable
        t = build_oracle_projection_problem(
            model_mga,
            poi_expressions,
            zO,
        )

        status = solve_network(network_mga)

        if spatial:
            zI = extract_diversified_capacity(target_techs, network_mga)[0]
        else:
            zI = extract_diversified_capacity(target_techs, network_mga)[1]

        # --------------------------------------------------------
        # STEP 4. Construct projection cut
        # --------------------------------------------------------

        sigma = extract_oracle_projection_cut_normal(
            model_mga,
            pois=mga_weights_neutral.index,
        )

        # Normalize dual vector
        sigma_norm_factor = np.linalg.norm(sigma.values)

        if not np.isfinite(sigma_norm_factor) or sigma_norm_factor == 0:
            raise ValueError("Invalid projection-cut normal.")

        sigma = -sigma / sigma_norm_factor

        # Ensure consistent variable ordering
        assert sigma.index.equals(points.index)
        assert zI.index.equals(points.index)

        rhs = sigma @ zI

        sigma_norm, rhs_norm = normalise_cut(
            sigma.values,
            rhs,
            lb.values,
            ub.values,
        )

        # Normalise projected and trial points
        zI_norm = normalise_point(
            zI.values,
            lb.values,
            ub.values,
        )

        zO_norm = normalise_point(
            zO.values,
            lb.values,
            ub.values,
        )

        # --------------------------------------------------------
        # STEP 5. Update inner and outer approximations
        # --------------------------------------------------------

        # Add the new projected point to the inner approximation.
        inner_points_norm = np.vstack(
            [
                inner_points_norm,
                zI_norm,
            ]
        )

        # Additional geometric consistency check.
        assert sigma @ zO > sigma @ zI

        cut_tolerance = 1e-4

        outer.A = np.vstack(
            [
                outer.A,
                sigma_norm,
            ]
        )

        outer.b = np.append(
            outer.b,
            rhs_norm + cut_tolerance,
        )

        # Check that the updated outer approximation remains feasible.
        res = linprog(
            c=np.zeros(outer.A.shape[1]),
            A_ub=outer.A,
            b_ub=outer.b,
            bounds=[(-1, 1)] * outer.A.shape[1],
            method="highs",
        )

        print("Updated outer approximation feasible:", res.success)
        # The current cache corresponds to the vertices before the
        # latest cut and the inner approximation before the latest point.
        # It will be checked against this new point next iteration.
        previous_vertex_cache = current_vertex_cache

        pending_inner_point = zI_norm.copy()

        mga_spatial_alternatives[iteration], mga_alternatives[iteration] = (
            extract_diversified_capacity(target_techs, network_mga)
        )

        save_network(export_nc, iteration, network_mga)

        iteration += 1

    # ------------------------------------------------------------
    # Convert results to DataFrames
    # ------------------------------------------------------------

    mga_spatial_alternatives = alternatives_dict_to_frame(mga_spatial_alternatives)

    mga_alternatives = alternatives_dict_to_frame(mga_alternatives)

    return mga_spatial_alternatives, mga_alternatives  # , mga_weights
