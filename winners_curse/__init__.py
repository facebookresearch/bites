# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""BITES: Bayesian Inference for Treatment Effect Studies - Open Source Release"""

from .dist_params import GaussianDistParams
from .exp_utils import (
    _sum_gaussian_rvs,
    daily_variance,
    local_shrinkage_factor,
    pred_power_replication,
    predictive_daily_variance,
)
from .gaussian_model import GaussianModel

__all__ = [
    "GaussianDistParams",
    "GaussianModel",
    "_sum_gaussian_rvs",
    "daily_variance",
    "local_shrinkage_factor",
    "pred_power_replication",
    "predictive_daily_variance",
]
