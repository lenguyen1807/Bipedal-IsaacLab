This project implements reinforcement learning-based bipedal locomotion for the [Skyentific Poclegs humanoid robot](https://github.com/SkyentificGit/BipedalRobotSim) using NVIDIA Isaac Lab and RSL-RL.

## Description

The Skyentific Poclegs is a bipedal humanoid robot with the following joint configuration:
- **10 Degree-of-Freedom total** (5 per leg)
  - Hip Roll (HR)
  - Hip Abduction/Adduction (HAA)
  - Hip Flexion/Extension (HFE)
  - Knee Flexion/Extension (KFE)
  - Foot/Ankle Flexion/Extension (FFE)

## Project Structure

```
skyentific_poclegs/
├── skyentific_poclegs/
│   ├── assets/                          # Robot URDF/USD files
│   │   ├── robots/
│   │   │   └── poclegs.usd              # Robot model
│   │   └── skyentific_poclegs.py        # Robot configuration
│   └── tasks/
│       └── locomotion/
│           └── velocity/
│               ├── config/
│               │   └── skyentific_poclegs/
│               │       ├── agents/
│               │       │   └── rsl_rl_cfg.py    # PPO algorithm config
│               │       ├── rough_env_cfg.py     # Environment config
│               │       └── __init__.py          # Gym registration
│               └── mdp/
│                   ├── rewards.py               # Custom reward functions
│                   └── curriculums.py           # Curriculum learning
├── scripts/
│   ├── train.py                         # Training script
│   └── play.py                          # Evaluation script
```

## Installation

>[!IMPORTANT]
> Later

## Usage

### Training

Basic training with default settings:
```bash
python scripts/train.py --headless
```

Training with custom parameters:
```bash
python scripts/train.py \
    --headless \
    --num_envs 8192 \
    --max_iterations 30000 \
    --seed 42 \
    --run_name my_experiment
```

Training with video recording:
```bash
python scripts/train.py \
    --video \
    --video_interval 2000 \
    --video_length 200
```

### Evaluation

Play trained policy:
```bash
python scripts/play.py \
    --checkpoint logs/rsl_rl/skyentific_poclegs_rough/model_30000.pt
```

Record evaluation video:
```bash
python scripts/play.py \
    --checkpoint logs/rsl_rl/skyentific_poclegs_rough/model_30000.pt \
    --video \
    --video_length 1000
```

Export policy to ONNX/TorchScript:
```bash
python scripts/play.py \
    --checkpoint logs/rsl_rl/skyentific_poclegs_rough/model_30000.pt \
    --export
```

### Resuming Training

```bash
python scripts/train.py --resume
```

## 🎯 Features

### Environment
- **Rough terrain** with multiple sub-terrains:
  - Flat ground (30%)
  - Pyramid slopes (20%)
  - Stairs (10%)
  - Wave terrain (20%)
  - Random rough terrain (20%)
- **Velocity tracking commands** for forward/backward and turning motion
- **Progressive curriculum** for terrain difficulty

### Rewards
- **Velocity tracking**: Exponential rewards for following commanded velocities
- **Feet air time**: Encourages proper step height
- **Feet slide penalty**: Discourages dragging feet
- **Orientation control**: Maintains upright posture
- **Energy efficiency**: Penalizes excessive joint torques
- **Joint regularization**: Keeps joints near default positions

### Domain Randomization
- **Physics properties**: Friction, restitution, mass
- **Initial states**: Random poses and velocities at reset
- **External disturbances**: Random pushes during episodes

### Curriculum Learning
- **Terrain progression**: Automatically increases difficulty based on performance
- **Push strength**: Gradually increases external perturbations
- **Velocity commands**: Expands velocity range as tracking improves

## Training Details

### Default Hyperparameters
- **Algorithm**: PPO (Proximal Policy Optimization)
- **Parallel environments**: 4096
- **Steps per environment**: 24
- **Training iterations**: 30,000
- **Learning rate**: 1e-3 (adaptive schedule)
- **Network architecture**: [512, 256, 128] (actor & critic)
- **Discount factor (γ)**: 0.99
- **GAE lambda (λ)**: 0.95

### Observations (Policy Input)
1. Base linear velocity (3D)
2. Base angular velocity (3D)
3. Projected gravity (3D)
4. Velocity commands (3D)
5. Hip joint positions (2)
6. Knee/hip joint positions (6)
7. Foot joint positions (2)
8. Joint velocities (10)
9. Last actions (10)

**Total: 42 dimensions**

## 📈 Monitoring Training

Training logs are saved to `logs/rsl_rl/skyentific_poclegs_rough/`. View with TensorBoard:

```bash
tensorboard --logdir logs/rsl_rl/skyentific_poclegs_rough
```

Key metrics to monitor:
- `Loss/policy`: Policy loss (should decrease)
- `Loss/value`: Value function loss (should decrease)
- `Policy/mean_reward`: Average episode reward (should increase)
- `Train/mean_step_time`: Steps per second (training speed)

## 🔧 Customization

### Modifying Rewards
Edit `skyentific_poclegs/tasks/locomotion/velocity/config/skyentific_poclegs/rough_env_cfg.py`:

```python
@configclass
class RewardsCfg:
    # Adjust weights
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,  # Increase to emphasize velocity tracking
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
```

### Adding Custom Rewards
1. Define function in `skyentific_poclegs/tasks/locomotion/velocity/mdp/rewards.py`
2. Add term to `RewardsCfg` in `rough_env_cfg.py`

### Changing Network Architecture
Edit `skyentific_poclegs/tasks/locomotion/velocity/config/skyentific_poclegs/agents/rsl_rl_cfg.py`:

```python
policy = RslRlPpoActorCriticCfg(
    actor_hidden_dims=[256, 128, 64],  # Smaller network
    critic_hidden_dims=[256, 128, 64],
    activation="elu",
)
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named skyentific_poclegs`
```bash
# Solution: Install package in editable mode
cd skyentific_poclegs
pip install -e .
```

**Issue**: `FileNotFoundError: poclegs.usd not found`
```bash
# Solution: Ensure USD file exists
ls skyentific_poclegs/assets/robots/poclegs.usd
```

**Issue**: Training is very slow
```bash
# Solution: Reduce number of environments
python scripts/train.py --num_envs 2048
```

**Issue**: Robot falls immediately
```bash
# Solution: Check joint limits and initial pose
# Verify in: skyentific_poclegs/assets/skyentific_poclegs.py
```

## 🔗 Resources

- [Isaac Lab Documentation](https://isaac-sim.github.io/IsaacLab/)
- [RSL-RL GitHub](https://github.com/leggedrobotics/rsl_rl)
- [Skyentific YouTube](https://www.youtube.com/@skyentific)
