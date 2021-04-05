# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# The code is based on HigherHRNet-Human-Pose-Estimation.
# (https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation)
# ------------------------------------------------------------------------------

from .transforms import Compose
from .transforms import RandomAffineTransform
from .transforms import ToTensor
from .transforms import Normalize
from .transforms import RandomHorizontalFlip

from .build import build_transforms
from .build import FLIP_CONFIG
