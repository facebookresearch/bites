# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""Simulation generation utilities for BITES."""

import numpy as np

from .dist_params import GaussianDistParams


def effect_gen_gaussian(design_params: GaussianDistParams) -> float:
    """
    Generate an effect size from a Gaussian design prior based on
    the parameters in design_params.

    Args:
        design_params(GaussianDistParams): parameters of the design
        prior from which to draw a value.

    Returns:
        A float of the effect drawn from the gaussian prior.
    """
    return float(np.random.normal(design_params.mean, design_params.variance**0.5))
