# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Skyentific Poclegs bipedal robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import SKYENTIFIC_ASSETS_DIR

##
# Robot Configuration
##

SKYENTIFIC_POCLEGS_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{SKYENTIFIC_ASSETS_DIR}/robots/poclegs.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.449),  # Initial height of 44.9 cm
        joint_pos={
            # Left and Right Hip Roll
            "LL_HR": 0.0,
            "LR_HR": 0.0,
            # Hip Abduction/Adduction
            "LL_HAA": -0.1745,  # -10 degrees
            "LR_HAA": -0.1745,
            # Hip Flexion/Extension
            "LL_HFE": -0.1745,  # -10 degrees
            "LR_HFE": -0.1745,
            # Knee Flexion/Extension
            "LL_KFE": 0.3491,  # 20 degrees
            "LR_KFE": 0.3491,
            # Foot/Ankle Flexion/Extension
            "LL_FFE": -0.1745,  # -10 degrees
            "LR_FFE": -0.1745,
        },
    ),
    actuators={
        # Hip Roll actuators
        "hr": DelayedPDActuatorCfg(
            joint_names_expr=[".*HR"],
            effort_limit=24.0,
            velocity_limit=23.0,
            stiffness=10.0,
            damping=1.5,
            armature=6.9e-5 * 81,
            friction=0.02,
            min_delay=0,  # Min: 0ms (2.0ms * 0)
            max_delay=4,  # Max: 8ms (2.0ms * 4)
        ),
        # Hip Abduction/Adduction actuators
        "haa": DelayedPDActuatorCfg(
            joint_names_expr=[".*HAA"],
            effort_limit=30.0,
            velocity_limit=15.0,
            stiffness=15.0,
            damping=1.5,
            armature=9.4e-5 * 81,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
        # Hip and Knee Flexion/Extension actuators
        "kfe": DelayedPDActuatorCfg(
            joint_names_expr=[".*HFE", ".*KFE"],
            effort_limit=30.0,
            velocity_limit=20.0,
            stiffness=15.0,
            damping=1.5,
            armature=1.5e-4 * 81,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
        # Foot/Ankle Flexion/Extension actuators
        "ffe": DelayedPDActuatorCfg(
            joint_names_expr=[".*FFE"],
            effort_limit=20.0,
            velocity_limit=23.0,
            stiffness=10.0,
            damping=1.5,
            armature=6.9e-5 * 81,
            friction=0.02,
            min_delay=0,
            max_delay=4,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
