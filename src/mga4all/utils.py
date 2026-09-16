# Functions that are useful across various parts of the code

import pandas as pd


def alternatives_dict_to_frame(alternatives_dict: dict):
    df = pd.concat(
        (
            pd.DataFrame(alternatives_dict[x]).fillna(0).round(2).sum(axis=1)
            for x in alternatives_dict
        ),
        axis=1,
    )

    return df
