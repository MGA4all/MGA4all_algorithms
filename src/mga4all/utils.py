# Functions that are useful across various parts of the code

import numpy as np
import pandas as pd
import pypsa


def alternatives_dict_to_frame(alternatives_dict: dict):
    df = pd.DataFrame(alternatives_dict).fillna(0).round(2)
    return df


def save_network(export_nc: str | bool, iteration: int, network_mga: pypsa.Network):
    """Export the given network as netcdf file.

    Args:
        export_nc: Prefix for netcdf file. If boolean False, saving is skipped.
            can be boolean True to save without prefix (equivalent to passing '')
        iteration: iteration number of the algorithm in progress
        network_mga: PyPSA network to export
    """
    if export_nc is False:
        pass
    elif isinstance(export_nc, str):
        network_mga.export_to_netcdf(f"{export_nc}mga_{iteration}.nc")
    else:
        network_mga.export_to_netcdf(f"mga_{iteration}.nc")


def normalise_point(z, lb, ub):
    return 2.0 * (z - lb) / (ub - lb) - 1.0


def denormalise_point(z_norm, lb, ub):
    return 0.5 * (ub - lb) * (z_norm + 1.0) + lb


def normalise_l2(weights_series):
    weights_series = weights_series / np.sqrt((weights_series**2).sum())
    return weights_series
