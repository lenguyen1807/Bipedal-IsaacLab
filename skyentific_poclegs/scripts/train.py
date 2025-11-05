#!/usr/bin/env python3
# Copyright (c) 2022-2024, The Berkeley Humanoid Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play/evaluate trained policy for Skyentific Poclegs robot."""

import argparse
import os

import gymnasium as gym
import torch

from isaaclab.app import AppLauncher

# Parse command line arguments
parser = argparse.ArgumentParser(description="Play trained RL policy.")
parser.add_argument(
    "--video",
    action="store_true",
    default=False,
    help="Record evaluation video.",
)
parser.add_argument(
    "--video_length",
    type=int,
    default=1000,
    help="Length of recorded video (in steps).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=50,
    help="Number of environments to visualize.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Velocity-Rough-Skyentific-Poclegs-Play-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint file (.pt). If not provided, will search in logs.",
)
parser.add_argument(
    "--export",
    action="store_true",
    default=False,
    help="Export policy to ONNX and TorchScript.",
)

# Add Isaac Lab launcher arguments
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Force enable cameras for video recording
if args.video:
    args.enable_cameras = True

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Import after Isaac Sim is initialized
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import (
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg


def main():
    """Main evaluation function."""

    # Parse environment configuration
    env_cfg = parse_env_cfg(
        args.task,
        num_envs=args.num_envs,
        use_fabric=not args.cpu,
    )

    # Get RL algorithm configuration
    agent_cfg = gym.spec(args.task).kwargs["rsl_rl_cfg_entry_point"]
    agent_cfg = agent_cfg()  # Instantiate

    # Determine checkpoint path
    if args.checkpoint:
        resume_path = args.checkpoint
    else:
        # Search in logs directory
        log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
        resume_path = get_checkpoint_path(
            log_root_path,
            agent_cfg.load_run,
            agent_cfg.load_checkpoint,
        )

    if not os.path.exists(resume_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {resume_path}\n"
            f"Please provide a valid checkpoint path using --checkpoint or ensure "
            f"training logs exist in: {log_root_path}"
        )

    log_dir = os.path.dirname(resume_path)
    print(f"[INFO] Loading checkpoint from: {resume_path}")
    print(f"[INFO] Log directory: {log_dir}")

    # Create environment
    env = gym.make(
        args.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args.video else None,
    )

    # Wrap environment for video recording
    if args.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args.video_length,
            "disable_logger": True,
        }
        print(f"[INFO] Recording video to: {video_kwargs['video_folder']}")
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Wrap environment for RSL-RL
    env = RslRlVecEnvWrapper(env)

    # Create PPO runner (for loading policy)
    runner = OnPolicyRunner(
        env,
        agent_cfg.to_dict(),
        log_dir=None,  # Don't create new logs
        device=agent_cfg.device,
    )

    # Load trained model
    print("[INFO] Loading trained policy...")
    runner.load(resume_path)

    # Get inference policy
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # Export policy if requested
    if args.export:
        print("[INFO] Exporting policy...")
        export_model_dir = os.path.join(log_dir, "exported")
        os.makedirs(export_model_dir, exist_ok=True)

        # Export to TorchScript
        export_policy_as_jit(
            runner.alg.actor_critic,
            normalizer=runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.pt",
        )
        print(f"[INFO] Exported TorchScript to: {export_model_dir}/policy.pt")

        # Export to ONNX
        export_policy_as_onnx(
            runner.alg.actor_critic,
            normalizer=runner.obs_normalizer,
            path=export_model_dir,
            filename="policy.onnx",
        )
        print(f"[INFO] Exported ONNX to: {export_model_dir}/policy.onnx")

    # Print evaluation info
    print("\n" + "=" * 80)
    print(f"Evaluating Policy: {args.task}")
    print("=" * 80)
    print(f"Number of environments: {env_cfg.scene.num_envs}")
    print(f"Checkpoint: {resume_path}")
    print("=" * 80 + "\n")

    # Run evaluation
    print("[INFO] Starting evaluation...")
    print("[INFO] Press Ctrl+C to stop.")

    obs, _ = env.get_observations()
    timestep = 0

    try:
        while simulation_app.is_running():
            with torch.inference_mode():
                # Get actions from policy
                actions = policy(obs)

                # Step environment
                obs, _, _, _ = env.step(actions)

                timestep += 1

                # Print progress every 100 steps
                if timestep % 100 == 0:
                    print(f"[INFO] Timestep: {timestep}")

    except KeyboardInterrupt:
        print("\n[INFO] Evaluation stopped by user.")

    print(f"[INFO] Evaluation complete! Total timesteps: {timestep}")

    # Cleanup
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] Evaluation failed with error: {e}")
        raise
    finally:
        # Ensure simulation is closed
        if "simulation_app" in globals():
            simulation_app.close()
