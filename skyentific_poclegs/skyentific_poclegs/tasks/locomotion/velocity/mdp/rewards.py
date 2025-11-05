# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for bipedal locomotion."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float,
) -> torch.Tensor:
    """Reward longer steps using L2-kernel.

    Encourages the robot to lift its feet and take proper steps. The reward is based
    on the time feet spend in the air, clamped between min and max thresholds.

    Args:
        env: The learning environment.
        command_name: Name of the velocity command.
        sensor_cfg: Configuration for the contact sensor.
        threshold_min: Minimum air time to start rewarding (in seconds).
        threshold_max: Maximum air time to cap rewards (in seconds).

    Returns:
        Reward tensor for each environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get first contact events and last air time
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[
        :, sensor_cfg.body_ids
    ]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]

    # Calculate air time above minimum threshold
    air_time = (last_air_time - threshold_min) * first_contact

    # Clamp to maximum threshold
    air_time = torch.clamp(air_time, max=threshold_max - threshold_min)

    # Sum across all feet
    reward = torch.sum(air_time, dim=1)

    # Only reward when robot is commanded to move
    command = env.command_manager.get_command(command_name)
    reward *= torch.norm(command[:, :2], dim=1) > 0.1

    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold_min: float,
    threshold_max: float,
) -> torch.Tensor:
    """Reward single-stance phase for bipeds.

    Encourages alternating single-leg support, which is essential for bipedal walking.
    Rewards the minimum time spent in either contact or air when only one foot is grounded.

    Args:
        env: The learning environment.
        command_name: Name of the velocity command.
        sensor_cfg: Configuration for the contact sensor.
        threshold_min: Minimum time in single stance to reward (in seconds).
        threshold_max: Maximum time to cap rewards (in seconds).

    Returns:
        Reward tensor for each environment.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    # Get current air and contact times
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0

    # Time in current mode (air or contact)
    in_mode_time = torch.where(in_contact, contact_time, air_time)

    # Check if exactly one foot is in contact (single stance)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1

    # Get minimum time across both feet during single stance
    reward = torch.min(
        torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1
    )[0]

    # Clamp to maximum and apply minimum threshold
    reward = torch.clamp(reward, max=threshold_max)
    reward *= reward > threshold_min

    # Only reward when commanded to move
    command = env.command_manager.get_command(command_name)
    reward *= torch.norm(command[:, :2], dim=1) > 0.1

    return reward


def feet_slide(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding on the ground.

    Discourages the robot from dragging its feet, which wastes energy and
    can damage the robot.

    Args:
        env: The learning environment.
        sensor_cfg: Configuration for the contact sensor.
        asset_cfg: Configuration for the robot asset.

    Returns:
        Penalty tensor for each environment (higher = more sliding).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]

    # Check if feet are in contact (force > 1N)
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
        .norm(dim=-1)
        .max(dim=1)[0]
        > 1.0
    )

    # Get horizontal velocity of feet
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]

    # Penalize velocity when in contact
    penalty = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)

    return penalty
