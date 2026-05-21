import numpy as np
import pandas as pd


def build_foldaware_features(train_df, apply_df):
    """
    Build fold-aware aggregation features.

    train_df:
        Fold training data only.
        Used to compute group statistics.

    apply_df:
        Data to transform.
        Can be train fold, validation fold, or test fold.
    """

    # TODO:
    # Move the existing team logic here:
    # - zip-level installment/freight features
    # - seller statistics
    # - category statistics
    # - customer statistics

    return apply_df, foldaware_cols


def zscore(series, mu=None, sigma=None):
    """
    Standardise a feature using mean and standard deviation.
    """

    # TODO:
    # Move existing zscore logic here

    return z


def build_motif_indices(df, z_stats=None, fit=True):
    """
    Build composite motif indices.

    fit=True:
        Compute z-score statistics from the training fold.

    fit=False:
        Reuse z-score statistics learned from the training fold.
    """

    # TODO:
    # Move existing motif logic here:
    # - frustration_index
    # - financial_burden_index
    # - consistency_index

    return df, z_stats


def build_features(train_df, apply_df, fit=True, z_stats=None):
    """
    Main function called inside cross-validation.

    For train fold:
        transformed_train, z_stats = build_features(train_df, train_df, fit=True)

    For validation/test fold:
        transformed_valid, _ = build_features(train_df, valid_df, fit=False, z_stats=z_stats)
    """

    transformed_df, foldaware_cols = build_foldaware_features(
        train_df,
        apply_df
    )

    transformed_df, z_stats = build_motif_indices(
        transformed_df,
        z_stats=z_stats,
        fit=fit
    )

    return transformed_df, z_stats

""" usage example:
from Scripts.inter_feature_builder import build_features

train_fe, z_stats = build_features(
    train_df,
    train_df,
    fit=True
)

valid_fe, _ = build_features(
    train_df,
    valid_df,
    fit=False,
    z_stats=z_stats
)
"""