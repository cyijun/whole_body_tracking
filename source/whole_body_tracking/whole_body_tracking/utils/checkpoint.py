"""Compatibility helpers for RSL-RL checkpoints."""

from __future__ import annotations

import torch
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def is_legacy_rsl_rl_checkpoint(checkpoint: Mapping[str, Any]) -> bool:
    """Return whether a checkpoint uses the pre-RSL-RL-4 actor-critic layout."""
    return "model_state_dict" in checkpoint and "actor_state_dict" not in checkpoint


def convert_legacy_rsl_rl_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    """Convert an ActorCritic checkpoint to the RSL-RL 5 actor/critic layout.

    The old actor and critic MLPs map one-to-one to the new ``MLPModel.mlp``
    modules. Observation normalizers and the state-independent policy standard
    deviation are moved into the actor and critic model state dictionaries.
    """
    if not is_legacy_rsl_rl_checkpoint(checkpoint):
        return dict(checkpoint)

    model_state = checkpoint["model_state_dict"]
    actor_state: dict[str, torch.Tensor] = {}
    critic_state: dict[str, torch.Tensor] = {}

    for key, value in model_state.items():
        if key == "std":
            actor_state["distribution.std_param"] = value
        elif key.startswith("actor."):
            actor_state[f"mlp.{key.removeprefix('actor.')}"] = value
        elif key.startswith("critic."):
            critic_state[f"mlp.{key.removeprefix('critic.')}"] = value

    for key, value in checkpoint.get("obs_norm_state_dict", {}).items():
        actor_state[f"obs_normalizer.{key}"] = value
    for key, value in checkpoint.get("privileged_obs_norm_state_dict", {}).items():
        critic_state[f"obs_normalizer.{key}"] = value

    if not any(key.startswith("mlp.") for key in actor_state):
        raise ValueError("Legacy checkpoint does not contain actor MLP weights.")
    if not any(key.startswith("mlp.") for key in critic_state):
        raise ValueError("Legacy checkpoint does not contain critic MLP weights.")

    legacy_state_keys = {"model_state_dict", "obs_norm_state_dict", "privileged_obs_norm_state_dict"}
    converted = {key: value for key, value in checkpoint.items() if key not in legacy_state_keys}
    converted.update(
        actor_state_dict=actor_state,
        critic_state_dict=critic_state,
        iter=checkpoint.get("iter", 0),
        infos=checkpoint.get("infos"),
        legacy_checkpoint_converted=True,
    )
    return converted


def load_compatible_checkpoint(
    runner: Any,
    path: str | Path,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
) -> dict | None:
    """Load either a current or legacy checkpoint into an RSL-RL 5 runner."""
    checkpoint = torch.load(path, weights_only=False, map_location=map_location)
    was_legacy = is_legacy_rsl_rl_checkpoint(checkpoint)
    checkpoint = convert_legacy_rsl_rl_checkpoint(checkpoint)

    if was_legacy and load_cfg is None:
        load_cfg = {
            "actor": True,
            "critic": True,
            "optimizer": "optimizer_state_dict" in checkpoint,
            "iteration": True,
            "rnd": False,
        }

    load_iteration = runner.alg.load(checkpoint, load_cfg, strict)
    if load_iteration:
        runner.current_learning_iteration = checkpoint.get("iter", 0)
    return checkpoint.get("infos")


def convert_checkpoint_file(source: str | Path, destination: str | Path) -> Path:
    """Convert a legacy checkpoint on disk, preserving current checkpoints."""
    source_path = Path(source)
    destination_path = Path(destination)
    if source_path.resolve() == destination_path.resolve():
        raise ValueError("Source and destination checkpoint paths must be different.")
    checkpoint = torch.load(source_path, weights_only=False, map_location="cpu")
    converted = convert_legacy_rsl_rl_checkpoint(checkpoint)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination_path)
    return destination_path
