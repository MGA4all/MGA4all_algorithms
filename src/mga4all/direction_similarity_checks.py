import pandas as pd
import pypsa
import numpy as np
from pandas.api.types import is_number
from scipy.stats import spearmanr


def ranking_similarity(weights_a, weights_b):
    rank_a = weights_a.argsort().argsort()
    rank_b = weights_b.argsort().argsort()
    corr, pvalue = spearmanr(rank_a, rank_b)
    return corr


def angular_similarity(weights_a, weights_b):
    return float((weights_a * weights_b).sum())


def perturb_rank_flip(
    weights_series, flip_strength=0.25, min_positive=1e-6, normalise=True
):
    values = weights_series.to_numpy().copy()

    eligible = np.where(values >= 0)[0]

    if len(eligible) < 2:
        return weights_series.copy()

    eligible_values = values[eligible]
    sorted_local = np.argsort(eligible_values)

    rank_index = np.random.randint(len(sorted_local) - 1)

    index_i = eligible[sorted_local[rank_index]]
    index_j = eligible[sorted_local[rank_index + 1]]

    gap = values[index_j] - values[index_i]
    delta = gap / 2 + flip_strength * np.random.rand()

    new_i = values[index_i] + delta
    new_j = values[index_j] - delta

    if new_j <= min_positive:
        delta = values[index_j] - min_positive
        new_i = values[index_i] + delta
        new_j = min_positive

    values[index_i] = new_i
    values[index_j] = new_j

    perturbed = pd.Series(values, index=weights_series.index, name=weights_series.name)

    if normalise:
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed /= norm

    return perturbed


def perturb_noise(weights_series, noise_scale=0.05, normalise=True):
    values = weights_series.to_numpy().copy()
    values += np.random.normal(0, noise_scale, len(values))
    perturbed = pd.Series(values, index=weights_series.index, name=weights_series.name)

    if normalise:
        norm = np.linalg.norm(perturbed)
        if norm > 0:
            perturbed /= norm

    return perturbed


def is_different_enough(
    new_weights, stored_weights, cosine_threshold=0.97, ranking_threshold=0.9
):
    stored_weights = list(stored_weights.values())

    if len(stored_weights) == 0:
        return True

    for old in stored_weights:
        cosine_similarity = angular_similarity(new_weights, old)
        corr_ranking = ranking_similarity(new_weights, old)

        if cosine_similarity > cosine_threshold:
            return False

        if corr_ranking > ranking_threshold:
            return False

    return True
