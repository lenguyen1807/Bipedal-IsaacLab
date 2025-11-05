# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum functions for progressive difficulty scaling."""

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Curriculum for terrain difficulty based on robot progress.

    Increases terrain difficulty when robot walks far enough, decreases when
    robot fails to meet velocity command requirements.

    Args:
        env: The learning environment.
        env_ids: Environment indices to update.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Mean terrain level across all environments.
    """
    asset = env.scene[asset_cfg.name]
    terrain = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    # Calculate distance walked from spawn point
    distance = torch.norm(
        asset.data.root_pos_w[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1
    )

    # Progress to harder terrain if walked more than half the terrain size
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2

    # Regress to easier terrain if walked less than half of commanded distance
    expected_distance = (
        torch.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    )
    move_down = (distance < expected_distance) & ~move_up

    # Update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)

    return torch.mean(terrain.terrain_levels.float())


def modify_push_force(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: int = 0,
) -> float:
    """Curriculum for external push perturbations.

    Gradually increases push strength as robot becomes more stable, decreases
    if robot falls too frequently.

    Args:
        env: The learning environment.
        env_ids: Environment indices (unused, affects all).
        term_name: Name of the push event term.
        max_velocity: Maximum push velocity [x, y] in m/s.
        interval: Steps between curriculum updates.
        starting_step: Step to start applying curriculum.

    Returns:
        Current push velocity setting.
    """
    # Check if push term exists
    try:
        term_cfg = env.event_manager.get_term_cfg("push_robot")
    except KeyError:
        return 0.0

    curr_setting = term_cfg.params["velocity_range"]["x"][1]

    # Don't modify before starting step or between intervals
    if env.common_step_counter < starting_step:
        return curr_setting
    if env.common_step_counter % interval != 0:
        return curr_setting

    # Count terminations
    base_contacts = torch.sum(env.termination_manager._term_dones["base_contact"])
    time_outs = torch.sum(env.termination_manager._term_dones["time_out"])

    # Increase difficulty if robot is stable (few early terminations)
    if base_contacts < time_outs * 2:
        curr_setting = np.clip(curr_setting * 1.5, 0.0, max_velocity[0])

    # Decrease difficulty if robot falls too often
    elif base_contacts > time_outs / 2:
        curr_setting = np.clip(curr_setting - 0.2, 0.0, max_velocity[0])

    # Update push velocity range
    term_cfg.params["velocity_range"]["x"] = (-curr_setting, curr_setting)
    term_cfg.params["velocity_range"]["y"] = (-curr_setting, curr_setting)
    env.event_manager.set_term_cfg("push_robot", term_cfg)

    return curr_setting


def modify_command_velocity(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    max_velocity: Sequence[float],
    interval: int,
    starting_step: int = 0,
) -> float:
    """Curriculum for commanded velocity.

    Increases velocity commands as robot learns to track them well.

    Args:
        env: The learning environment.
        env_ids: Environment indices to check.
        term_name: Name of velocity tracking reward term.
        max_velocity: Maximum velocity range [min, max] in m/s.
        interval: Steps between curriculum updates.
        starting_step: Step to start applying curriculum.

    Returns:
        Current maximum commanded velocity.
    """
    command_cfg = env.command_manager.get_term("base_velocity").cfg
    curr_lin_vel_x = command_cfg.ranges.lin_vel_x

    # Don't modify before starting step or between intervals
    if env.common_step_counter < starting_step:
        return curr_lin_vel_x[1]
    if env.common_step_counter % interval != 0:
        return curr_lin_vel_x[1]

    # Check tracking performance
    term_cfg = env.reward_manager.get_term_cfg(term_name)
    episode_rewards = env.reward_manager._episode_sums[term_name][env_ids]
    avg_reward_per_step = torch.mean(episode_rewards) / env.max_episode_length

    # Increase velocity range if tracking well (>80% of max reward)
    target_reward = 0.8 * term_cfg.weight * env.step_dt
    if avg_reward_per_step > target_reward:
        curr_lin_vel_x = (
            np.clip(curr_lin_vel_x[0] - 0.5, max_velocity[0], 0.0),
            np.clip(curr_lin_vel_x[1] + 0.5, 0.0, max_velocity[1]),
        )
        command_cfg.ranges.lin_vel_x = curr_lin_vel_x

    return curr_lin_vel_x[1]
