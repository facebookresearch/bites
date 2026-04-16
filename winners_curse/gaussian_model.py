# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Gaussian-Gaussian conjugate Bayesian model for BITES."""

from __future__ import annotations

import sys

import numpy as np
import scipy.stats

from .bayes_model import BayesianModel
from .data_gen import data_gen_gaussian
from .dist_params import GaussianDistParams
from .exp_utils import (
    _calc_ratio_mean,
    _calc_ratio_var,
    _calc_required_mean_for_prob_failure,
    _calc_required_mean_for_prob_success,
    _calc_required_obs_mean,
)
from .sim_gen import effect_gen_gaussian


class GaussianModel(BayesianModel):
    """This class implements a Bayesian model for a Gaussian-Gaussian conjugate model.
    theta sim N(prior_mean, prior_var)
    x_t sim N(theta, data_variance)

    See §2.5 of Bayesian Data Analysis by Gelman et al (3rd edition)
    """

    def __init__(
        self,
        data_mean: float,
        data_variance: float,
        n_obs: int,
        prior_params: GaussianDistParams | None = None,
        shrinkage_g: float | None = 1.0,
    ) -> None:
        """Instantiates the model object with the minimal set of values required to
        estimate the posterior distribution.

        Args:
            data_mean: the cumulative mean of the observations.
            data_variance: the sampling variance.
            n_obs: the number of observations (e.g., how many periods have elapsed).
            prior_params: (optional) an instance of GaussianDistParams specifying the prior parameters.
            By default set to an uninformative prior.
            shrinkage_g: (optional) the 'g' hyperparameter that controls the amount of shrinkage
            induced by the prior. By default set to 1.0 (i.e. no influence on the specified prior_params).
        """
        # pyrefly: ignore [missing-attribute]
        super().__init__()

        self._data_mean = data_mean
        self._data_variance = data_variance
        self._n_obs = n_obs
        self._prior_params = prior_params or GaussianDistParams(0.0, sys.float_info.max)
        self._shrinkage_g: float = shrinkage_g if shrinkage_g is not None else 1.0
        self._prior_params = self._calc_g_prior(self._shrinkage_g)
        self._posterior_params: GaussianDistParams | None = None

    @classmethod
    def from_posterior(
        cls,
        posterior_params: GaussianDistParams,
        data_variance: float = 1.0,
    ) -> GaussianModel:
        """Alternative approach for instantiating the model object directly from a set of provided posterior
        parameters. In this case the only additional parameter required is the variance (required for
        predictive distributions).

        Args:
            posterior_params: an instance of GaussianDistParams specifying the posterior parameters
            data variance: the sampling variance.
        """
        model = cls.__new__(cls)
        # pyrefly: ignore [missing-attribute]
        super(GaussianModel, model).__init__()
        model._posterior_params = posterior_params
        model._data_variance = data_variance
        return model

    @property
    def data_mean(self):
        return self._data_mean

    @property
    def data_variance(self):
        return self._data_variance

    @property
    def n_obs(self) -> int:
        return self._n_obs

    @property
    def prior_params(self):
        return self._prior_params

    @property
    def shrinkage_g(self) -> float:
        return self._shrinkage_g

    @property
    def posterior_params(self) -> GaussianDistParams | None:
        return self._posterior_params

    @data_mean.setter
    def data_mean(self, value: float) -> None:
        self._data_mean = value
        self._posterior_params = None

    @data_variance.setter
    def data_variance(self, value: float) -> None:
        self._data_variance = value
        self._posterior_params = None

    @n_obs.setter
    def n_obs(self, value):
        self._n_obs = value
        self._posterior_params = None

    @prior_params.setter
    def prior_params(self, value: GaussianDistParams) -> None:
        self._prior_params = value
        self._prior_params = self._calc_g_prior(self._shrinkage_g)
        self._posterior_params = None

    @shrinkage_g.setter
    def shrinkage_g(self, value: float) -> None:
        self._shrinkage_g = value
        self._prior_params = self._calc_g_prior(self._shrinkage_g)
        self._posterior_params = None

    @posterior_params.setter
    def posterior_params(self, value):
        self._posterior_params = value

    def calc_posterior(self) -> GaussianDistParams:
        """Calculates the posterior mean and variance for a Gaussian-Gaussian model.

        Returns:
            An instance of GaussianDistParams specifying the posterior parameters.
        """
        if not self.posterior_params:
            post_var = 1 / (
                1 / self.prior_params.variance + self.n_obs / self.data_variance
            )
            post_mean = post_var * (
                self.prior_params.mean / self.prior_params.variance
                + (self.n_obs / self.data_variance) * self.data_mean
            )
            self.posterior_params = GaussianDistParams(
                mean=post_mean, variance=post_var
            )

        assert self.posterior_params is not None
        return self.posterior_params

    def calc_posterior_central_credible_intervals(
        self,
        probability_interval: float,
    ) -> tuple[float, float]:
        """Calculates a central credible interval that contains a specified posterior density.

        Args:
            probability_interval: the probability mass contained in the interval.

        Returns:
            A tuple containing the upper and lower credible interval bounds.
        """
        post_params = self.calc_posterior()
        # pyrefly: ignore [missing-attribute]
        post_pdf = scipy.stats.norm(
            loc=post_params.mean, scale=np.sqrt(post_params.variance)
        )

        return (
            float(post_pdf.ppf((1 - probability_interval) / 2)),
            float(post_pdf.ppf(probability_interval + (1 - probability_interval) / 2)),
        )

    def calc_posterior_predictive(self, n_additional_obs: int) -> GaussianDistParams:
        """Calculates the mean and variance of the posterior predictive distribution.

        theta sim N(prior_mean, prior_var)
        x_t sim N(theta, data_variance)

        The probability of a future observation tilde{y} given y is:

        P(tilde{y}|y) = int P(tilde{y}|theta) P(theta|y) dtheta

        The mean of the posterior predictive distribution is simply the mean of
        the posterior distribution. The variance of the posterior
        predictive distribution has two components: the posterior uncertainty
        in theta and the sampling uncertainty due to data_variance.

        See §2.5 of Bayesian Data Analysis by Gelman et al (3rd edition)

        Args:
            n_additional_obs: the number of future observations.

        Returns:
            An instance of GaussianDistParams specifying the posterior predictive parameters.
        """
        post_params = self.calc_posterior()
        return GaussianDistParams(
            mean=post_params.mean,
            variance=post_params.variance + self.data_variance / n_additional_obs,
        )

    def calc_predictive_intervals(
        self, n_additional_obs: int, probability_interval: float
    ) -> tuple[float, float]:
        """Calculates a predictive interval based on the posterior predictive distribution.

        Args:
            n_additional_obs: the number of future observations.
            probability_interval: probability bounds for the predictive interval.

        Returns:
            A tuple containing the upper and lower predictive interval bounds.
        """
        post_pred_params = self.calc_posterior_predictive(n_additional_obs)
        # pyrefly: ignore [missing-attribute]
        post_pred_pdf = scipy.stats.norm(
            loc=post_pred_params.mean, scale=np.sqrt(post_pred_params.variance)
        )

        return (
            float(post_pred_pdf.ppf((1 - probability_interval) / 2)),
            float(
                post_pred_pdf.ppf(probability_interval + (1 - probability_interval) / 2)
            ),
        )

    def calc_pred_prob_of_success(
        self, n_additional_obs: int, threshold: float, prob: float
    ) -> float:
        """Calculates the predictive probability of success for `n_additional_obs` future observations.

        "success" is define as P(treatment_effect > threshold) > prob.

        The calculation proceeds as follows:

        1. Calculate the posterior mean and variance of the treatment effect.
        2. Calculate the future expected posterior variance after collecting `n_additional_obs`.
        3. Calculate the required future posterior mean to achieve success.
        4. Calculate the required raw mean to obtain the required future posterior mean. (That is,
            undo the shrinkage from the prior.)
        5. Calculate the posterior predictive mean and variance of the treatment effect for
            `n_additional_obs` additional observations.
        6. Calculate the probability that the raw mean drawn from N(pred_mean, pred_var) will
            exceed the required value for success.

        Args:
            n_additional_obs: the number of future observations.
            threshold: the threshold above which the outcome is a success.
            prob: the desired probability that the treatment effect is above the threshold (e.g., 0.95).
        Returns:
            A float specifying the predictive probability of success.
        """
        posterior_params = self.calc_posterior()
        posterior_predictive_params = self.calc_posterior_predictive(
            n_additional_obs=n_additional_obs
        )
        future_post_var = 1 / (
            1 / posterior_params.variance + n_additional_obs / self.data_variance
        )
        required_post_mean = _calc_required_mean_for_prob_success(
            variance=future_post_var, threshold=threshold, prob=prob
        )
        required_obs_mean = _calc_required_obs_mean(
            desired_post_mean=required_post_mean,
            n_obs=n_additional_obs,
            prior_params=posterior_params,
            data_variance=self.data_variance,
        )
        # pyrefly: ignore [missing-attribute]
        return scipy.stats.norm.sf(
            (required_obs_mean - posterior_predictive_params.mean)
            / np.sqrt(posterior_predictive_params.variance)
        )

    def calc_curr_prob_of_success(self, threshold: float) -> float:
        """This function calculates the probability of success based on the current data.

        which is defined as P(treatment_effect > threshold).

        The calculation proceeds as follows:

        1. Calculate the posterior mean and variance of the treatment effect.
        2. Calculate the posterior probability that we have a treatment effect that exceeds some threshold.

        Note that success here is defined differently from the predictive probability notion. This result is
        based solely on current data.
        Args:
            threshold: the threshold above which the outcome is a success.
        Returns:
            A float specifying the probability of success.
        """

        posterior_params = self.calc_posterior()
        # pyrefly: ignore [missing-attribute]
        return scipy.stats.norm.sf(
            x=threshold,
            loc=posterior_params.mean,
            scale=np.sqrt(posterior_params.variance),
        )

    def calc_pred_prob_of_failure(
        self, n_additional_obs: int, threshold: float, prob: float
    ) -> float:
        """Calculates the predictive probability of failure for `n_additional_obs` future observations.

        "failure" is define as P(treatment_effect < threshold) > prob.

        The calculation proceeds as follows:

        1. Calculate the posterior mean and variance of the treatment effect.
        2. Calculate the future expected posterior variance after collecting `n_additional_obs`.
        3. Calculate the required future posterior mean to fail.
        4. Calculate the required raw mean to obtain the required future posterior mean. (That is,
        undo the shrinkage from the prior.)
        5. Calculate the posterior predictive mean and variance of the treatment effect for
        `n_additional_obs` additional observations.
        6. Calculate the probability that the raw mean drawn from N(pred_mean, pred_var) will
        fall below the required value for failure.

        Args:
            n_additional_obs: the number of future observations.
            threshold: the threshold above which the outcome is a success.
            prob: the desired probability that the treatment effect is above the threshold (e.g., 0.95).

        Returns:
            A float specifying the predictive probability of failure.
        """
        posterior_params = self.calc_posterior()
        posterior_predictive_params = self.calc_posterior_predictive(
            n_additional_obs=n_additional_obs
        )

        future_post_var = 1 / (
            1 / posterior_params.variance + n_additional_obs / self.data_variance
        )
        required_post_mean = _calc_required_mean_for_prob_failure(
            variance=future_post_var,
            threshold=threshold,
            prob=prob,
        )
        required_obs_mean = _calc_required_obs_mean(
            desired_post_mean=required_post_mean,
            n_obs=n_additional_obs,
            prior_params=posterior_params,
            data_variance=self.data_variance,
        )
        # pyrefly: ignore [missing-attribute]
        return scipy.stats.norm.cdf(
            (required_obs_mean - posterior_predictive_params.mean)
            / np.sqrt(posterior_predictive_params.variance)
        )

    def _calc_g_prior(
        self,
        g: float,
    ) -> GaussianDistParams:
        """Calculates prior parameters according to Zellener's g-prior approach for a
        Gaussian-Gaussian conjugate model.

        The strength of the prior is determined by a single hyperparameter, g, that specifies the weight
        of the prior relative to the data: lower values of g place higher weight on the prior, and as g
        increases the weight on the prior decreases.

        Args:
            prior_mean: prior belief about the mean parameter
            prior_var: prior belief about the variance parameter
            g : coefficient g that specifies the relative weight of information contained in the prior vs the data

        Returns:
            An instance of GaussianDistParams specifying the prior parameters.
        """
        return GaussianDistParams(
            mean=self.prior_params.mean,
            variance=g * self.prior_params.variance,
        )

    def sim_gaussian_oc(
        self,
        n_iter: int,
        data_obs: int,
        prob: float,
        design_params: GaussianDistParams,
        control_params: GaussianDistParams,
    ) -> tuple[float, float]:
        """Function that calculates the fpr/ fdr based on operating characteristics as
        described in Zhao et al (Section 2.3: "ON BAYESIAN SEQUENTIAL CLINICAL TRIAL DESIGNS").

        Args:
           design_params(GaussianDistParams): parameters of the design
            prior from which to draw a value of the effect.
           control_params(GaussianDistParams): parameters of the data generating
            process for the control group. Mean here should be 1.0.
           n_iter(int): number of iterations for which to run the
            simulation.
           data_obs(int): number of observations in the simulated dataset.
           prob(float): the probability with which we want to ensure success (e.g., 0.95).
        Returns:
           A float of the FPR/ FDR as specified in eq 7 on the above.
        """
        is_negative = 0.0
        is_rejected = 0.0
        is_false_rejection = 0.0

        for _ in range(n_iter):
            # Generate the effect and log its sign.
            sim_effect = effect_gen_gaussian(design_params)

            if sim_effect < 0:
                is_negative += 1.0

            # Generate the data and find the relevant moments.
            sim_data = data_gen_gaussian(sim_effect, data_obs, control_params)
            sim_data_mean = _calc_ratio_mean(
                sim_data["y_treatment"], sim_data["y_control"]
            )

            sim_data_var = _calc_ratio_var(
                np.mean(sim_data["y_treatment"]),
                np.var(sim_data["y_treatment"]),
                np.mean(sim_data["y_control"]),
                np.var(sim_data["y_control"]),
            )

            # Instantiate the model (within this iteration)
            sim_model = GaussianModel(
                data_mean=sim_data_mean,
                data_variance=sim_data_var,
                n_obs=self.n_obs,
                prior_params=self.prior_params,
            )

            sim_interval = sim_model.calc_posterior_central_credible_intervals(
                probability_interval=prob
            )

            # Assess the decision.
            if sim_interval[0] > 0:
                is_rejected += 1.0
            if (sim_effect < 0) and (sim_interval[0] > 0):
                is_false_rejection += 1.0
        fdr = is_false_rejection / max(is_rejected, 1.0)
        fpr = is_false_rejection / max(is_negative, 1.0)
        return (fpr, fdr)
