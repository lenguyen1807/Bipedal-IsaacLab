# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tasks for Skyentific Poclegs robot."""

from isaaclab_tasks.utils import import_packages

# Blacklist certain packages from being imported
_BLACKLIST_PKGS = ["utils"]

# Recursively import all task configurations
import_packages(__name__, _BLACKLIST_PKGS)
