# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Distribution parameter classes for BITES."""

from abc import ABC
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistParams(ABC):
    """This is an abstract base class that serves as a parent class for all distribution
    classes. Currently the only practical use of this is for type hints, but in future
    we might add abstract member functions to enforce consistent implementation of
    operations on distributions.
    """

    pass


@dataclass(frozen=True)
class GaussianDistParams(DistParams):
    """This data class stores the parameters of a Gaussian distribution.

    Attributes:
        mean (float): Mean of the Gaussian distribution.
        variance (float): Variance of the Gaussian distribution.
    """

    mean: float
    variance: float

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GaussianDistParams):
            return NotImplemented
        return np.allclose([self.mean, self.variance], [other.mean, other.variance])
