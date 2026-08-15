"""Replay a converted motion with Isaac Lab.

.. code-block:: bash

    # Usage
    python scripts/replay_npz.py --motion_file motion.npz --max_steps 500
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import os
import pathlib
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
motion_source = parser.add_mutually_exclusive_group(required=True)
motion_source.add_argument("--registry_name", type=str, help="W&B motion artifact name.")
motion_source.add_argument("--motion_file", type=str, help="Local motion NPZ path.")
parser.add_argument("--max_steps", type=int, default=None, help="Stop after this many rendered frames.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    if args_cli.motion_file is not None:
        motion_file = os.path.abspath(args_cli.motion_file)
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Motion file does not exist: {motion_file}")
    else:
        import wandb

        registry_name = args_cli.registry_name
        if ":" not in registry_name:
            registry_name += ":latest"
        artifact = wandb.Api().artifact(registry_name)
        motion_file = str(pathlib.Path(artifact.download()) / "motion.npz")

    motion = MotionLoader(
        motion_file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
    )
    expected_fps = 1.0 / sim_dt
    if not np.isclose(motion.fps, expected_fps):
        raise ValueError(f"Motion is {motion.fps:g} Hz, but replay is configured for {expected_fps:g} Hz.")
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    frame_count = 0

    # Simulation loop
    while simulation_app.is_running():
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_pose[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_velocity = robot.data.default_root_vel.torch.clone()
        root_velocity[:, :3] = motion.body_lin_vel_w[time_steps][:, 0]
        root_velocity[:, 3:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_link_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_com_velocity_to_sim_index(root_velocity=root_velocity)
        robot.write_joint_position_to_sim_index(position=motion.joint_pos[time_steps])
        robot.write_joint_velocity_to_sim_index(velocity=motion.joint_vel[time_steps])
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_pose[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        frame_count += 1
        if args_cli.max_steps is not None and frame_count >= args_cli.max_steps:
            break
        time_steps += 1
        time_steps[time_steps >= motion.time_step_total] = 0


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
