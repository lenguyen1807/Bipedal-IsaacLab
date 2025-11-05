# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for rough terrain locomotion."""

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.envs import ViewerCfg

from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
)
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

# Import custom MDP functions and robot config
import skyentific_poclegs.tasks.locomotion.velocity.mdp as skyentific_mdp
from skyentific_poclegs.assets import SKYENTIFIC_POCLEGS_CFG

##
# Terrain Configuration
##

ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.3,
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1,
            slope_range=(0.0, 0.4),
            platform_width=2.0,
            border_width=0.25,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.0, 0.1),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "wave_terrain": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.0, 0.2),
            num_waves=4,
            border_width=0.25,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.0, 0.06),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)

##
# Environment Configuration
##


@configclass
class SkyentificPoclegsRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """Configuration for Skyentific Poclegs rough terrain locomotion."""

    @configclass
    class ObservationsCfg:
        """Observation configuration."""

        @configclass
        class PolicyCfg(ObsGroup):
            """Observations for the policy."""

            # Proprioceptive observations
            base_lin_vel = ObsTerm(
                func=mdp.base_lin_vel,
                noise=Unoise(n_min=-0.1, n_max=0.1),
            )
            base_ang_vel = ObsTerm(
                func=mdp.base_ang_vel,
                noise=Unoise(n_min=-0.2, n_max=0.2),
            )
            projected_gravity = ObsTerm(
                func=mdp.projected_gravity,
                noise=Unoise(n_min=-0.05, n_max=0.05),
            )

            # Commands
            velocity_commands = ObsTerm(
                func=mdp.generated_commands,
                params={"command_name": "base_velocity"},
            )

            # Joint states (grouped by function)
            hip_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.03, n_max=0.03),
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR"])},
            )
            kfe_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.05, n_max=0.05),
                params={
                    "asset_cfg": SceneEntityCfg(
                        "robot", joint_names=[".*HAA", ".*HFE", ".*KFE"]
                    )
                },
            )
            ffe_pos = ObsTerm(
                func=mdp.joint_pos_rel,
                noise=Unoise(n_min=-0.08, n_max=0.08),
                params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*FFE"])},
            )
            joint_vel = ObsTerm(
                func=mdp.joint_vel_rel,
                noise=Unoise(n_min=-1.5, n_max=1.5),
            )

            # Previous actions
            actions = ObsTerm(func=mdp.last_action)

            # Height scan (disabled for now)
            height_scan = None

            def __post_init__(self):
                self.enable_corruption = True
                self.concatenate_terms = True

        policy: PolicyCfg = PolicyCfg()

    @configclass
    class RewardsCfg:
        """Reward configuration."""

        # Task rewards
        track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp,
            weight=1.0,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )
        track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp,
            weight=0.5,
            params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
        )

        # Bipedal locomotion rewards
        feet_air_time = RewTerm(
            func=skyentific_mdp.feet_air_time,
            weight=2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
                "command_name": "base_velocity",
                "threshold_min": 0.2,
                "threshold_max": 0.5,
            },
        )
        feet_slide = RewTerm(
            func=skyentific_mdp.feet_slide,
            weight=-0.25,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*ffe"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*ffe"),
            },
        )

        # Regularization penalties
        lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
        ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
        joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
        action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

        # Contact penalties
        undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[".*hfe", ".*haa"]
                ),
                "threshold": 1.0,
            },
        )

        # Joint deviation penalties
        joint_deviation_hip = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.1,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*HR", ".*HAA"])
            },
        )
        joint_deviation_knee = RewTerm(
            func=mdp.joint_deviation_l1,
            weight=-0.01,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*KFE"])},
        )

        # Stability penalties
        flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)
        dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    @configclass
    class TerminationsCfg:
        """Termination configuration."""

        time_out = DoneTerm(func=mdp.time_out, time_out=True)
        base_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"),
                "threshold": 1.0,
            },
        )

    @configclass
    class EventCfg:
        """Randomization events."""

        # Startup randomization
        physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.2, 1.25),
                "dynamic_friction_range": (0.2, 1.25),
                "restitution_range": (0.0, 0.1),
                "num_buckets": 64,
            },
        )
        scale_all_link_masses = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "mass_distribution_params": (0.9, 1.1),
                "operation": "scale",
            },
        )
        add_base_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "mass_distribution_params": (-1.0, 1.0),
                "operation": "add",
            },
        )

        # Reset randomization
        reset_base = EventTerm(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "yaw": (-3.14, 3.14),
                },
                "velocity_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (-0.5, 0.5),
                    "roll": (-0.5, 0.5),
                    "pitch": (-0.5, 0.5),
                    "yaw": (-0.5, 0.5),
                },
            },
        )
        reset_robot_joints = EventTerm(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "position_range": (0.5, 1.5),
                "velocity_range": (0.0, 0.0),
            },
        )

        # Interval randomization (external disturbances)
        push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(10.0, 15.0),
            params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
        )

    @configclass
    class CurriculumCfg:
        """Curriculum configuration."""

        terrain_levels = CurrTerm(func=skyentific_mdp.terrain_levels_vel)

        push_force_levels = CurrTerm(
            func=skyentific_mdp.modify_push_force,
            params={
                "term_name": "push_robot",
                "max_velocity": [3.0, 3.0],
                "interval": 200 * 24,
                "starting_step": 1500 * 24,
            },
        )

        command_vel = CurrTerm(
            func=skyentific_mdp.modify_command_velocity,
            params={
                "term_name": "track_lin_vel_xy_exp",
                "max_velocity": [-1.5, 3.0],
                "interval": 200 * 24,
                "starting_step": 5000 * 24,
            },
        )

    # Assign configurations
    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    # Viewer configuration
    viewer = ViewerCfg(
        eye=(3.5, 3.5, 0.5),
        origin_type="env",
        env_index=1,
        asset_name="robot",
    )

    def __post_init__(self):
        """Post-initialization to set up scene."""
        super().__post_init__()

        # Configure terrain
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ROUGH_TERRAINS_CFG,
            max_init_terrain_level=0,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        # Set robot configuration
        self.scene.robot = SKYENTIFIC_POCLEGS_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        # Disable height scanner (not using for now)
        self.scene.height_scanner = None


@configclass
class SkyentificPoclegsRoughEnvCfg_PLAY(SkyentificPoclegsRoughEnvCfg):
    """Configuration for playing/evaluating trained policies."""

    def __post_init__(self):
        super().__post_init__()

        # Smaller scene for visualization
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5

        # Spawn on all terrain levels
        self.scene.terrain.max_init_terrain_level = None

        # Reduce terrain variety for visualization
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable observation noise
        self.observations.policy.enable_corruption = False

        # Disable external disturbances
        self.events.push_robot = None
