"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--motion_file", type=str, default=None, help="Path to the motion file.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many simulation steps.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
    if not getattr(args_cli, "visualizer_explicit", False):
        args_cli.visualizer = ["kit"]
    elif args_cli.visualizer is None or "kit" not in args_cli.visualizer:
        parser.error("--video requires the Kit visualizer; use --viz kit or omit --viz.")

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import pathlib
import re
import torch
from importlib import metadata

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg,
    multi_agent_to_single_agent,
)
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Import extensions to set up environment tasks
import whole_body_tracking.tasks  # noqa: F401
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx
from whole_body_tracking.utils.my_on_policy_runner import MotionOnPolicyRunner as OnPolicyRunner


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.sim.use_fabric = not args_cli.disable_fabric

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)

    if args_cli.wandb_path:
        import wandb

        path_parts = args_cli.wandb_path.strip("/").split("/")
        if len(path_parts) not in (3, 4):
            raise ValueError("--wandb_path must be ENTITY/PROJECT/RUN_ID[/MODEL_FILE].")
        run_path = "/".join(path_parts[:3])
        requested_file = path_parts[3] if len(path_parts) == 4 else None
        api = wandb.Api()
        wandb_run = api.run(run_path)
        checkpoint_files = [entry.name for entry in wandb_run.files() if re.fullmatch(r"model_[0-9]+\.pt", entry.name)]
        if requested_file is not None:
            file = requested_file
        else:
            if not checkpoint_files:
                raise RuntimeError(f"W&B run '{run_path}' has no model_<iteration>.pt checkpoint.")
            file = max(checkpoint_files, key=lambda name: int(name.removeprefix("model_").removesuffix(".pt")))

        wandb_file = wandb_run.file(str(file))
        wandb_file.download("./logs/rsl_rl/temp", replace=True)

        print(f"[INFO]: Loading model checkpoint from: {run_path}/{file}")
        resume_path = f"./logs/rsl_rl/temp/{file}"

        if args_cli.motion_file is None:
            art = next((a for a in wandb_run.used_artifacts() if a.type == "motions"), None)
            if art is None:
                raise RuntimeError("The W&B run has no motion artifact; pass --motion_file explicitly.")
            env_cfg.commands.motion.motion_file = str(pathlib.Path(art.download()) / "motion.npz")

    else:
        print(f"[INFO] Loading experiment from directory: {log_root_path}")
        if args_cli.checkpoint is not None and os.path.isfile(args_cli.checkpoint):
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    if args_cli.motion_file is not None:
        motion_path = os.path.abspath(args_cli.motion_file)
        if not os.path.isfile(motion_path):
            raise FileNotFoundError(f"Motion file does not exist: {motion_path}")
        print(f"[INFO]: Using motion file from CLI: {motion_path}")
        env_cfg.commands.motion.motion_file = motion_path

    log_dir = os.path.dirname(resume_path)

    # Video recording is driven internally by Isaac Lab 3.x during env.step().
    if args_cli.video:
        video_dir = os.path.join(log_dir, "videos", "play")
        env_cfg.commands.motion.debug_vis = False
        env_cfg.scene.contact_forces.debug_vis = False
        env_cfg.video_recorders = [
            VideoRecorderCfg(
                source="visualizer:kit",
                output_dir=video_dir,
                output_filename_prefix="play",
                video_length=args_cli.video_length,
                video_interval=0,
            )
        ]
        print(f"[INFO] Recording {args_cli.video_length} steps to: {video_dir}")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")

    export_motion_policy_as_onnx(
        env.unwrapped,
        ppo_runner.alg.get_policy(),
        path=export_model_dir,
        filename="policy.onnx",
    )
    attach_onnx_metadata(env.unwrapped, args_cli.wandb_path if args_cli.wandb_path else "none", export_model_dir)
    # reset environment
    obs = env.get_observations()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, dones, _ = env.step(actions)
            policy.reset(dones)
        timestep += 1
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
