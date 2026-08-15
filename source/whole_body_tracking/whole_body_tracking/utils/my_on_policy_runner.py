import os

from rsl_rl.env import VecEnv
from rsl_rl.runners.on_policy_runner import OnPolicyRunner

import wandb
from whole_body_tracking.utils.checkpoint import load_compatible_checkpoint
from whole_body_tracking.utils.exporter import attach_onnx_metadata, export_motion_policy_as_onnx


class MyOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner with legacy checkpoint loading and W&B ONNX export."""

    def load(self, path: str, load_cfg=None, strict=True, map_location=None):
        return load_compatible_checkpoint(self, path, load_cfg, strict, map_location)

    def _export_to_wandb(self, checkpoint_path: str):
        if self.logger.logger_type != "WandbLogWriter" or wandb.run is None:
            return

        policy_path = os.path.dirname(checkpoint_path)
        filename = f"{os.path.basename(policy_path)}.onnx"
        export_motion_policy_as_onnx(
            self.env.unwrapped,
            self.alg.get_policy(),
            path=policy_path,
            filename=filename,
        )
        run_path = wandb.run.path
        if not isinstance(run_path, str):
            run_path = "/".join(run_path)
        attach_onnx_metadata(self.env.unwrapped, run_path, path=policy_path, filename=filename)
        wandb.save(os.path.join(policy_path, filename), base_path=policy_path)

    def save(self, path: str, infos=None):
        """Save the model, export ONNX, and upload it when W&B is active."""
        super().save(path, infos)
        self._export_to_wandb(path)


class MotionOnPolicyRunner(MyOnPolicyRunner):
    def __init__(
        self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device="cpu", registry_name: str = None
    ):
        super().__init__(env, train_cfg, log_dir, device)
        self.registry_name = registry_name

    def save(self, path: str, infos=None):
        """Save/export the policy and link its source motion artifact once."""
        super().save(path, infos)
        if self.logger.logger_type == "WandbLogWriter" and wandb.run is not None and self.registry_name is not None:
            wandb.run.use_artifact(self.registry_name)
            self.registry_name = None
