"""
MIT License

Copyright (c) 2025 HThawley
Original code sourced from: MH-MGA (https://github.com/HThawley/MH-MGA)

Modified by Francesco Lombardi, 2026
Changes: 
- Updated the functions to handle inputs in the format required by MGA4all project
- Updated some of the function descriptions

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import numpy as np
import pandas as pd

###
# Condensed metrics DataFrame--------------------
###


def exploration_metrics_progression(points, lb, ub):
    """
    Calculates diversity metrics as a progression over a set of MGA alternatives

    Parameters
    ----------
    points : pd.DataFrame
        Rows are decision variables and columns are MGA alternatives.
    lb : pd.Series
        Lower bound for each decision variable.
    ub : pd.Series
        Upper bound for each decision variable.
    """

    metrics_df = (
        pd.DataFrame(index=points.columns, columns=["Shannon", "VESA"])
        .astype(float)
        .fillna(0)
    )

    for alt in metrics_df.index:

        metrics_df.loc[alt, "Shannon"] = mean_of_shannon_of_projections(
            points.iloc[:, 0 : alt + 1], lb=lb, ub=ub
        )

        metrics_df.loc[alt, "VESA"] = volume_estimation_by_shadow_addition(
            points.iloc[:, 0 : alt + 1]
        )

    return metrics_df


###
# Shannon Index ----------------------------------
###


def mean_of_shannon_of_projections(points, lb, ub):
    """
    Mean Shannon index across each decision variable
    of a set of MGA alternatives.

    Parameters
    ----------
    points : pd.DataFrame
        Rows are decision variables and columns are MGA alternatives.
    lb : pd.Series
        Lower bound for each decision variable.
    ub : pd.Series
        Upper bound for each decision variable.
    """

    if not points.index.equals(lb.index) or not points.index.equals(ub.index):
        raise ValueError("points, lb, and ub must have identical indices.")

    npoint = points.shape[1]
    ndim = points.shape[0]

    nbin = max(2, int(npoint**0.5))

    acc = 0.0
    counts = np.zeros(nbin, dtype=int)

    for variable in points.index:
        counts[:] = 0

        acc += _shannon_index(
            points.loc[variable],
            lb.loc[variable],
            ub.loc[variable],
            nbin,
            counts,
        )

    acc /= np.log(nbin)
    acc /= ndim

    return acc


def _shannon_index(values, lb, ub, nbin, counts):
    bin_width = (ub - lb) / nbin

    if bin_width == 0:
        return 0.0

    for value in values:
        idx = int((value - lb) / bin_width)

        if idx < 0:
            idx = 0
        if idx > nbin - 1:
            idx = nbin - 1

        counts[idx] += 1

    npoint = len(values)

    H = 0.0
    nonzero = 0

    for count in counts:
        if count > 0:
            p = count / npoint
            H -= p * np.log(p)
            nonzero += 1

    # Miller-Madow bias correction
    H += (nonzero - 1) / (2.0 * npoint)

    return H


###
# VESA - Volume Estimation by Shadow Addition --------------
###


def volume_estimation_by_shadow_addition(points):
    """
    Estimate the volume of a set of points using
    Volume Estimation by Shadow Addition (VESA).

    Parameters
    ----------
    points : pd.DataFrame
        Rows are decision variables and columns are MGA alternatives.

    Returns
    -------
    float
        Sum of the convex-hull areas of all 2D projections.
    """
    vesa = 0.0

    variables = points.index

    for k, variable_1 in enumerate(variables):
        for variable_2 in variables[k + 1 :]:
            projection = np.stack(
                (
                    points.loc[variable_1].to_numpy(),
                    points.loc[variable_2].to_numpy(),
                ),
                axis=-1,
            )

            vesa += _convex_hull_area(projection)

    return vesa


def _convex_hull_area(points):
    """
    Calculates the area of the convex hull of a set of 2D points.

    Uses the Monotone Chain (Andrew's) algorithm to find the convex hull vertices,
    and then the Shoelace formula to calculate the area.

    Args:
        points (np.ndarray): A 2D NumPy array of shape (N, 2) representing
                             the N points, where each row is [x, y].

    Returns:
        float: The area of the convex hull. Returns 0.0 if there are fewer than 3 unique points.
    """
    hull_points = _convex_hull(points)

    if len(hull_points) < 3:
        return 0.0

    return _shoelace_area(hull_points, len(hull_points))


def _convex_hull(points):
    n = points.shape[0]

    indices = np.argsort(points[:, 0])
    points = points[indices]

    i = 0
    while i < n:
        j = i

        while j < n and points[j, 0] == points[i, 0]:
            j += 1

        if j - i > 1:
            slice_to_sort = points[i:j]
            indices = np.argsort(slice_to_sort[:, 1])
            points[i:j] = slice_to_sort[indices]

        i = j

    hull_points = np.empty((2 * n, 2), dtype=points.dtype)
    hull_idx = 0

    for p in points:
        while (
            hull_idx >= 2
            and _cross_product(
                hull_points[hull_idx - 2],
                hull_points[hull_idx - 1],
                p,
            )
            <= 0
        ):
            hull_idx -= 1

        hull_points[hull_idx] = p
        hull_idx += 1

    t = hull_idx + 1

    for p in points[n - 2 :: -1]:
        while (
            hull_idx >= t
            and _cross_product(
                hull_points[hull_idx - 2],
                hull_points[hull_idx - 1],
                p,
            )
            <= 0
        ):
            hull_idx -= 1

        hull_points[hull_idx] = p
        hull_idx += 1

    hull_points = hull_points[:hull_idx]

    if hull_idx < 3:
        return hull_points

    return hull_points


def _cross_product(p1, p2, p3):
    """
    Calculates the 2D cross product (z-component) of vectors p1p2 and p1p3.
    This determines the orientation of the triplet (p1, p2, p3).
    A positive value means a counter-clockwise turn (left turn).
    A negative value means a clockwise turn (right turn).
    A zero value means the points are collinear.
    """
    return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])


def _shoelace_area(points_buffered, num_points):
    """
    Calculates the area of a polygon given its vertices using the Shoelace formula.
    The points must be ordered (e.g., clockwise or counter-clockwise).
    """
    area = 0.0
    for i in range(num_points):
        j = (i + 1) % num_points
        area += points_buffered[i, 0] * points_buffered[j, 1]
        area -= points_buffered[j, 0] * points_buffered[i, 1]
    return abs(area) / 2.0
