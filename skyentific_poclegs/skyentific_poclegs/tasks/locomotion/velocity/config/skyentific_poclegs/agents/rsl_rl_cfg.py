# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""RSL-RL agent configuration for Skyentific Poclegs."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class SkyentificPoclegsRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for rough terrain locomotion."""

    # Training parameters
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 200
    experiment_name = "skyentific_poclegs_rough"
    empirical_normalization = False

    # Policy network architecture
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    # PPO algorithm parameters
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class SkyentificPoclegsFlatPPORunnerCfg(SkyentificPoclegsRoughPPORunnerCfg):
    """PPO runner configuration for flat terrain locomotion.

    Uses smaller networks since flat terrain is easier.
    """

    def __post_init__(self):
        super().__post_init__()

        # Fewer iterations needed for flat terrain
        self.max_iterations = 15000
        self.experiment_name = "skyentific_poclegs_flat"

        # Smaller networks for simpler task
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]
