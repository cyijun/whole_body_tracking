# BeyondMimic Motion Tracking Code

[![IsaacSim](https://img.shields.io/badge/IsaacSim-6.0.1-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-develop-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://docs.python.org/3/whatsnew/3.12.html)
[![Linux platform](https://img.shields.io/badge/platform-linux--aarch64%20%7C%20x86__64-orange.svg)](https://ubuntu.com/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[[Website]](https://beyondmimic.github.io/)
[[Arxiv]](https://arxiv.org/abs/2508.08241)
[[Video]](https://youtu.be/RS_MtKVIAzY)

## Overview

BeyondMimic is a versatile humanoid control framework that provides highly dynamic motion tracking with the
state-of-the-art motion quality on real-world deployment and steerable test-time control with guided diffusion-based
controllers.

This repo covers the motion tracking training in BeyondMimic. **You should be able to
train any sim-to-real-ready motion in the LAFAN1 dataset, without tuning any parameters**.

For sim-to-sim and sim-to-real deployment, please refer to
the [motion_tracking_controller](https://github.com/HybridRobotics/motion_tracking_controller).

### Alternative Implementations

- There is an alternative reproduction of BeyondMimic in [mjlab](https://github.com/mujocolab/mjlab), a new Isaac Lab-style manager API powered by MuJoCo-Warp for RL and robotics research. See the implementation [here](https://github.com/mujocolab/mjlab/blob/main/src/mjlab/tasks/tracking/tracking_env_cfg.py).

## Installation

- Install Isaac Sim 6.0.1 and the current Isaac Lab `develop` branch by following the
  [Isaac Lab installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
  This migrated branch targets Python 3.12 and RSL-RL 5.x. The previously documented Isaac Sim 4.5 / Isaac Lab 2.1
  stack remains useful only for running the unmodified upstream code on x86_64.

- Clone this repository separately from the Isaac Lab installation (i.e., outside the `IsaacLab` directory):

```bash
# Option 1: SSH
git clone git@github.com:HybridRobotics/whole_body_tracking.git

# Option 2: HTTPS
git clone https://github.com/HybridRobotics/whole_body_tracking.git
```

- Pull the robot description files from GCS

```bash
# Enter the repository
cd whole_body_tracking
# Rename all occurrences of whole_body_tracking (in files/directories) to your_fancy_extension_name
curl -L -o unitree_description.tar.gz https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz && \
tar -xzf unitree_description.tar.gz -C source/whole_body_tracking/whole_body_tracking/assets/ && \
rm unitree_description.tar.gz
```

- Create a repository-local environment and install the library with `uv`:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
# Install Isaac Sim and Isaac Lab into this environment first.
uv pip install -e source/whole_body_tracking
```

The configured DGX Spark workspace can be activated with `source scripts/activate_env.zsh`. This also applies the
aarch64 Isaac Sim runtime settings and non-secret W&B defaults; `WANDB_API_KEY` remains in the user's shell profile.

## Motion Tracking

### Motion Preprocessing & Registry Setup

In order to manage the large set of motions we used in this work, we leverage the WandB registry to store and load
reference motions automatically.
Note: The reference motion should be retargeted and use generalized coordinates only.

- Gather the reference motion datasets (please follow the original licenses), we use the same convention as .csv of
  Unitree's dataset

    - Unitree-retargeted LAFAN1 Dataset is available
      on [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)
    - Sidekicks are from [KungfuBot](https://kungfu-bot.github.io/)
    - Christiano Ronaldo celebration is from [ASAP](https://github.com/LeCAR-Lab/ASAP).
    - Balance motions are from [HuB](https://hub-robot.github.io/)


- Log in to your WandB account; access Registry under Core on the left. Create a new registry collection with the name "
  Motions" and artifact type "All Types".


- Convert retargeted motions to include the maximum coordinates information (body pose, body velocity, and body
  acceleration) via forward kinematics,

```bash
python scripts/csv_to_npz.py --input_file {motion_name}.csv --input_fps 30 --output_name {motion_name}
```

This writes a 50 Hz, XYZW-quaternion NPZ and uploads it to the W&B registry. Legacy NPZ files without a
`quat_order` field are treated as WXYZ and converted in memory when loaded.

- Test if the WandB registry works properly by replaying the motion in Isaac Sim:

```bash
python scripts/replay_npz.py --registry_name {your-organization}-org/wandb-registry-motions/{motion_name}
# Or replay a local file:
python scripts/replay_npz.py --motion_file motion.npz
```

- Debugging
    - Make sure to export WANDB_ENTITY to your organization name, not your personal username.
    - If /tmp folder is not accessible, modify csv_to_npz.py L319 & L326 to a temporary folder of your choice.

### Policy Training

- Train policy by the following command:

```bash
python scripts/rsl_rl/train.py --task=Tracking-Flat-G1-v0 \
--registry_name {your-organization}-org/wandb-registry-motions/{motion_name} \
--logger wandb --log_project_name {project_name} --run_name {run_name}
```

For an offline/local smoke run, replace `--registry_name ...` with `--motion_file motion.npz`. Current AppLauncher
starts headless by default; pass `--viz kit` when a desktop visualizer is wanted.

### Policy Evaluation

- Play the trained policy by the following command:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=2 --wandb_path={wandb-run-path}
```

To record one 200-step MP4 from a single environment, add `--video`. The migrated launcher automatically enables the
Kit recorder in headless mode and writes the clip under the checkpoint directory's `videos/play/` folder:

```bash
python scripts/rsl_rl/play.py --task=Tracking-Flat-G1-v0 --num_envs=1 \
--wandb_path={wandb-run-path} --video --video_length=200
```

The WandB run path can be located in the run overview. It follows the format {your_organization}/{project_name}/ along
with a unique 8-character identifier. Note that run_name is different from run_path.

Legacy RSL-RL checkpoints from the Isaac Lab 2.1 version are converted automatically when loaded. To produce a
permanent RSL-RL 5.x checkpoint without starting the simulator:

```bash
python scripts/rsl_rl/convert_checkpoint.py old_model.pt new_model.pt
```

## Code Structure

Below is an overview of the code structure for this repository:

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp`**
  This directory contains the atomic functions to define the MDP for BeyondMimic. Below is a breakdown of the functions:

    - **`commands.py`**
      Command library to compute relevant variables from the reference motion, current robot state, and error
      computations. This includes pose and velocity error calculation, initial state randomization, and adaptive
      sampling.

    - **`rewards.py`**
      Implements the DeepMimic reward functions and smoothing terms.

    - **`events.py`**
      Implements domain randomization terms.

    - **`observations.py`**
      Implements observation terms for motion tracking and data collection.

    - **`terminations.py`**
      Implements early terminations and timeouts.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py`**
  Contains the environment (MDP) hyperparameters configuration for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/agents/rsl_rl_ppo_cfg.py`**
  Contains the PPO hyperparameters for the tracking task.

- **`source/whole_body_tracking/whole_body_tracking/robots`**
  Contains robot-specific settings, including armature parameters, joint stiffness/damping calculation, and action scale
  calculation.

- **`scripts`**
  Includes utility scripts for preprocessing motion data, training policies, and evaluating trained policies.

This structure is designed to ensure modularity and ease of navigation for developers expanding the project.
