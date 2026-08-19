# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Statistical utility functions for BITES."""

import numpy as np
import pandas as pd
import scipy as sp

from .dist_params import GaussianDistParams


def _calc_required_mean_for_prob_success(
    variance: float, threshold: float, prob: float
) -> float:
    """If x is Gaussian-distributed with variance `variance`, what must the
    mean be such that P(x > threshold) > prob?

    Args:
        variance: the variance of the Gaussian distribution.
        threshold: the threshold above which the outcome is a success.
        prob: the desired probability that the treatment effect is above the threshold (e.g., 0.95).

    Returns:
        The required mean for success.
    """
    return threshold - np.sqrt(variance) * sp.stats.norm.ppf(1 - prob)


def _calc_required_mean_for_prob_failure(
    variance: float, threshold: float, prob: float
) -> float:
    """If x is Gaussian-distributed with variance `variance`, what must the
    mean be such that P(x < threshold) > prob?

    Args:
        variance: the variance of the Gaussian distribution.
        threshold: the threshold below which the outcome is a failure.
        prob: the desired probability that the treatment effect is below the threshold (e.g., 0.95).
    Returns:
        The required mean for failure.
    """
    return threshold - np.sqrt(variance) * sp.stats.norm.ppf(prob)


def _calc_required_obs_mean(
    desired_post_mean: float,
    n_obs: int,
    prior_params: GaussianDistParams,
    data_variance: float,
) -> float:
    """What value must the mean of n observations have such that the
    posterior mean equals desired_post_mean?

    Args:
        desired_post_mean: the desired posterior mean.
        n_obs: the number of observations.
        prior_params: the prior parameters.
    Returns:
        The required observed mean to obtain the desired posterior mean.
    """
    post_var = 1 / (1 / prior_params.variance + n_obs / data_variance)
    return (data_variance / n_obs) * (
        desired_post_mean / post_var - prior_params.mean / prior_params.variance
    )


def _calc_ratio_var(
    treatment_mean: float,
    treatment_variance: float,
    control_mean: float,
    control_variance: float,
) -> float:
    """
    Function that calculates the variance of the ratio based on the delta method.

    Args:
        treatment_mean(float): the mean of the treatment group.
        treatment_variance(float): the variance of the treatment group.
        control_mean(float): the mean of the control group.
        control_variance(float): the variance of the control group.
    Returns:
        A first order approximation of the variance of the ratio.
    """
    return ((treatment_mean / control_mean) ** 2) * (
        (treatment_variance / treatment_mean**2) + (control_variance / control_mean**2)
    )


def _calc_ratio_mean(
    treatment_outcomes: pd.Series, control_outcomes: pd.Series
) -> float:
    """
    Function that calculates the ratio effect based on outcomes

    Args:
        treatment_outcomes: a pandas series of outcomes for the treated group.
        control_outcomes: a pandas series of outcomes for the control group.

    Returns:
        a float of the treatment effect expressed as a ratio.
    """
    return (np.mean(treatment_outcomes) / np.mean(control_outcomes)) - 1


def _sum_gaussian_rvs(gaussian_dists: list[GaussianDistParams]) -> GaussianDistParams:
    """Helper function to return the distribution of a random variable that is a
    sum of other Gaussian-distributed random variables.

    Args:
        gaussian_dists: a list of GaussianDistParams instances.

    Returns:
        A GaussianDistParams instance that specifies the distribution of the sum.
    """
    return GaussianDistParams(
        sum([gaussian.mean for gaussian in gaussian_dists]),
        sum([gaussian.variance for gaussian in gaussian_dists]),
    )


def predictive_daily_variance(
    daily_variance: float, new_units: int, orig_units: int
) -> float:
    """Scale the daily variance for a different segment count
    Args:
        daily_variance: the daily variance of the lift estimate assuming the original segment count
        orig_units: the number of units (e.g., segments) in the original experiment (e.g., the pretest)
        new_units: the number of units in the new experiment (e.g., the back-test)
    Returns:
        The daily variance for the new segment count
    """
    return (daily_variance * orig_units) / new_units


def pred_power_replication(
    posterior_params: GaussianDistParams,
    data_variance: float,
    n_obs: int,
    threshold: float = 0.0,
    prob: float = 0.90,
) -> float:
    """
    This function estimates the predictive power, which is a retrospective measure of power.

    If "success" is defined as P(treatment_effect > threshold) > prob, what is the probability
    that a repeat experiment that compares the same treatment and control is a success?

    Args:
        posterior_params: the estimated posterior distribution from an initial experiment (or a cumulative posterior distribution resulting from a collection of experiments).
        data_variance: the sampling variance in the replicate experiment.
        n_obs: the number of days in the replicate experiment.
        threshold: the threshold above which the outcome is a success.
        prob: the desired probability that the treatment effect is above the threshold (e.g., 0.90 as default).
    Returns:
        The predictive power
    """

    ## The required mean in the replicate experiment meet the specified success criteria
    required_post_mean = _calc_required_mean_for_prob_success(
        data_variance / n_obs, threshold, prob
    )
    ## Calculate the posterior predictive distribution
    posterior_predictive_params = GaussianDistParams(
        mean=posterior_params.mean,
        variance=posterior_params.variance + data_variance / n_obs,
    )
    # Calculate the probability of success
    return sp.stats.norm.sf(
        (required_post_mean - posterior_predictive_params.mean)
        / np.sqrt(posterior_predictive_params.variance)
    )


def daily_variance(
    mean: float, ci_high: float, n_obs: int, prob: float = 0.95
) -> float:
    """
    Estimates the variance of the daily lift estimate in an experiment.
    Args:
        mean: The point estimate of the lift in the experiment (e.g., using the ratio estimator from Deltoid)
        ci_high: The upper bound of the 90% confidence interval for the lift in the experiment (e.g., using the ratio estimator from Deltoid)
        prob: The confidence level or posterior probability of interest (e.g., 0.90).
        n_obs: The number of days in the experiment
    Returns:
        The variance of the daily lift estimate in the experiment
    """
    return ((ci_high - mean) / sp.stats.norm.ppf(prob, 0.0, 1.0) * n_obs**0.5) ** 2


def local_shrinkage_factor(exp_mean: float, prior_variance: float) -> float:
    """
    Computes the experiment-specific "local" shrinkage factor that modulates the global shrinkage factor for this experiment.
    Args:
        exp_mean: The point estimate of the lift in the experiment (e.g., using the ratio estimator from Deltoid)
        prior_variance: The variance of the prior distribution for the lift, which is assumed to be Gaussian
    """
    return ((exp_mean - 1) ** 2 + prior_variance) / (4 * prior_variance)
