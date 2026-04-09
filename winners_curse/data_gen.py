# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Data generation utilities for BITES."""

import numpy as np
import pandas as pd

from .dist_params import GaussianDistParams


def data_gen_gaussian(
    true_effect: float, sim_obs: int, control_params: GaussianDistParams
) -> pd.DataFrame:
    """
    Conditional on an effect value generates a dataset of potential outcomes
    for the treated and control groups under Gaussian assumptions on the outcome.

    Args:
        true_effect(float): the causal effect.
        sim_obs(int): the number of simulated entries in the dataset.
        control_params(GaussianDisributionParams): the mean and variance of the
            potential outcomes.
    Returns:
       A pandas dataframe with treatment and control entries assumed to be a
       fully balanced design (i.e., same number of observations in the two groups)
    """
    y_control = np.random.normal(
        control_params.mean, control_params.variance**0.5, sim_obs
    )
    y_treatment = np.random.normal(
        (1.0 + true_effect) * control_params.mean, control_params.variance**0.5, sim_obs
    )
    return pd.DataFrame({"y_control": y_control, "y_treatment": y_treatment})
