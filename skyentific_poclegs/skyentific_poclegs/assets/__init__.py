# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Assets for Skyentific Poclegs robot."""

import os

# Asset directory path
SKYENTIFIC_ASSETS_DIR = os.path.abspath(os.path.dirname(__file__))

# Import robot configurations
from .skyentific_poclegs import SKYENTIFIC_POCLEGS_CFG

__all__ = ["SKYENTIFIC_POCLEGS_CFG", "SKYENTIFIC_ASSETS_DIR"]
