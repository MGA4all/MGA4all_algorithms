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
# Shannon Index ----------------------------------
###

def mean_of_shannon_of_projections(points, lb, ub):
    """
    Mean Shannon index across each decision variable
    of a set of MGA alternatives.
    
    Parameters
    ----------
    points : dict
        Values are decision variables and keys are MGA alternatives.
    lb : pd.Series
        Lower bound for each decision variable.
    ub : pd.Series
        Upper bound for each decision variable.
    """

    points = pd.concat((
        pd.DataFrame(points[x]).fillna(0).round(2).sum(axis=1) for x in points
    ), axis=1)
    
    if not points.index.equals(lb.index) or not points.index.equals(ub.index):
        raise ValueError(
            "points, lb, and ub must have identical indices."
        )

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
