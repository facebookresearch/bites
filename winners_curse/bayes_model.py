# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Abstract base class for Bayesian models."""

from abc import ABC, abstractmethod

import pandas as pd

from .dist_params import DistParams


class BayesianModel(ABC):
    """This abstract base class should be the base class for all model classes to ensure
    a consistent interface. It establishes a number of methods that must be implemented by
    subclasses.
    """

    @abstractmethod
    def __init__(self) -> None:
        """Initialize the BayesianModel."""
        pass

    @abstractmethod
    def calc_posterior(self) -> DistParams:
        pass

    @abstractmethod
    def calc_posterior_central_credible_intervals(
        self,
        probability_interval: float,
    ) -> tuple[float, float] | tuple[pd.Series, pd.Series]:
        pass

    @abstractmethod
    def calc_posterior_predictive(self, n_additional_obs: int) -> DistParams:
        pass

    @abstractmethod
    def calc_predictive_intervals(
        self,
        n_additional_obs: int,
        probability_interval: float,
    ) -> tuple[float, float] | tuple[pd.Series, pd.Series]:
        pass

    @abstractmethod
    def calc_pred_prob_of_success(
        self, n_additional_obs: int, threshold: float, prob: float
    ) -> float | pd.Series:
        pass

    @abstractmethod
    def calc_curr_prob_of_success(self, threshold: float) -> float | pd.Series:
        pass

    @abstractmethod
    def calc_pred_prob_of_failure(
        self, n_additional_obs: int, threshold: float, prob: float
    ) -> float | pd.Series:
        pass
