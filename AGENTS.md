# Whole Body Tracking Agent Notes

## Scope

These instructions apply to the entire repository.

## Host platform

- This workspace is on NVIDIA GB10 / DGX Spark class hardware (`aarch64`) with Ubuntu 24.04 and CUDA 13.
- The original upstream stack used Python 3.10, Isaac Sim 4.5.0, and Isaac Lab 2.1.0. That stack has no native DGX Spark/aarch64 distribution and must not be installed on this host.
- The native host environment uses Python 3.12, Isaac Sim 6.0.1.0, CUDA 13 PyTorch, and the recorded Isaac Lab `develop` checkout. The source has been migrated to this stack and runtime-validated.

## Python environment

- Use the repository-local virtual environment through the project activation helper:

  ```bash
  source scripts/activate_env.zsh
  ```

- Do not install project dependencies into the system Python, Conda base, or the parent `dev` environment.
- On aarch64, direct Python commands that import Isaac Sim must preload the system GNU OpenMP library:

  ```bash
  export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1
  ```

- The user accepted the NVIDIA Omniverse EULA on 2026-08-15. `scripts/activate_env.zsh` therefore sets `OMNI_KIT_ACCEPT_EULA=yes` for non-interactive Isaac Sim startup.
- Isaac Lab is kept under `.deps/IsaacLab`; `.deps/` and `.venv/` are local-only and ignored by Git.

## Installed dependency stack

- Python: 3.12.x (`aarch64`)
- Isaac Sim: 6.0.1.0
- PyTorch: 2.11.0 with CUDA 13.0 wheels
- Isaac Lab: `develop`, pinned to the commit recorded below after installation
- RL framework: RSL-RL 5.4.1, selected by that Isaac Lab checkout
- W&B client: 0.19.11; this stays on the project's declared 0.19 floor and remains compatible with Isaac Sim's Click pin
- Video capture: MoviePy 1.0.3 with imageio-ffmpeg 0.6.0
- Project package: editable install from `source/whole_body_tracking`

Do not independently upgrade Isaac Sim, PyTorch, Isaac Lab, or RSL-RL. Update the complete compatibility set together and record the new versions here.

## Version strategy

- The exact README stack (Isaac Sim 4.5, Isaac Lab 2.1, Python 3.10) is x86_64-only and cannot run natively on this DGX Spark. Containers do not change CPU architecture, and emulating the x86_64 Kit/CUDA plugins is not a supported GPU path.
- The lowest-migration native alternative is Isaac Sim 5.1.0, Isaac Lab 2.3.1, Python 3.11, CUDA 13 PyTorch, and RSL-RL 3.0.1. It was researched but not installed because it is another transitional stack and cannot use this checkout's migrated XYZW/ProxyArray assumptions unchanged.
- Isaac Sim 6.0.1 / Isaac Lab `develop` is the supported route for this checkout. Do not mix files or packages between the old and new stacks; use a separate checkout and environment if reproducing the legacy baseline.

## Project runtime constraints

- The G1 asset must exist at `source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf`.
- Motion CSV input must follow the Unitree G1 joint order and is normally converted to 50 Hz. Motion phase advances from the stored NPZ `fps` and environment step time, so the 25 Hz low-frequency task advances two frames per control step instead of slowing a 50 Hz motion down.
- Training and replay accept exactly one of `--registry_name` and `--motion_file`. Registry artifacts must contain `motion.npz`.
- W&B public-cloud defaults are `WANDB_ENTITY=cyijun2k`, `WANDB_PROJECT=beyondmimic`, and `WANDB_BASE_URL=https://api.wandb.ai`. The API key is read from the user's `~/.zshrc` and must never be copied into repository files, logs, or command output.
- The key is exported by interactive zsh startup. Non-interactive automation does not necessarily source `~/.zshrc`; use an already initialized shell or `zsh -ic 'source scripts/activate_env.zsh && ...'`. The activation helper warns, but local-only commands remain available, when the key is absent.
- `WBT_WANDB_RUN_PATH=cyijun2k/beyondmimic/lfh2rk27` points to the latest successful prior run found during setup. `WBT_WANDB_MODEL_PATH=cyijun2k/beyondmimic/lfh2rk27/model_199.pt` selects its final uploaded checkpoint explicitly. `WBT_MOTION_ARTIFACT=cyijun2k/csv_to_npz/walk1_subject1_full:latest` points to its verified motion artifact. These are non-secret convenience variables defined by `scripts/activate_env.zsh` for `--wandb_path` and `--registry_name`; update them intentionally when promoting a newer run, model, or artifact.
- Pass `--logger wandb --log_project_name beyondmimic` when training. A play command needs a concrete run path such as `cyijun2k/beyondmimic/<run-id>`; the project URL is not a run path.
- After the API migration and EULA acceptance, the intended smoke command shape is:

  ```bash
  python scripts/rsl_rl/train.py \
    --task Tracking-Flat-G1-v0 \
    --registry_name "$WBT_MOTION_ARTIFACT" \
    --logger wandb \
    --log_project_name "$WANDB_PROJECT" \
    --num_envs 16 \
    --max_iterations 2
  ```

  The corresponding prior-run playback parameter is `--wandb_path "$WBT_WANDB_MODEL_PATH"`.

- Never begin with the configured 4096 environments and 30000 iterations. Use at most `--num_envs 16 --max_iterations 2` for the first smoke run unless the user explicitly requests a full run.
- Current AppLauncher is headless by default and no longer accepts `--headless`; pass `--viz kit` for the desktop visualizer.
- `--video` uses Isaac Lab 3.x `VideoRecorderCfg`, automatically selects the Kit visualizer when no `--viz` is given,
  and records through `env.step()` instead of Gymnasium's deprecated `render_mode="rgb_array"` path.
- Do not run simulation, training, or a long-lived replay loop unless explicitly requested. Bounded `--max_steps` replay/play and short `--max_iterations` smoke runs are preferred for validation.

## Migration compatibility

- Runtime code uses the Isaac Lab 3.x configclass, noise, physics/visualizer, XYZW quaternion, ProxyArray `.torch`, built-in COM randomization, and current math APIs.
- Newly generated motion files declare `quat_order=xyzw`. Old files with no declaration are assumed to be WXYZ and converted in memory with a warning.
- Agent configs use RSL-RL 5.x actor/critic models and TensorDict observation groups. `MotionOnPolicyRunner` supports both current checkpoints and legacy `model_state_dict` checkpoints; `scripts/rsl_rl/convert_checkpoint.py` performs standalone conversion.
- The old `model_199.pt` actor was numerically compared with its migrated representation and produced exact deterministic action parity (`max_abs_error=0.0`).
- Isaac Sim 6.0's URDF importer emits messages saying the URDF joints have no gain parameters. Isaac Lab then applies the configured actuator gains, as confirmed by the runtime joint table.
- Isaac Sim 6.0's URDF importer requires an explicit `ros_package_paths` mapping for G1's
  `package://unitree_description/...` mesh URLs. Without it, physics imports but the rendered robot is invisible.
- The current nested URDF geometry/contact resolver emits duplicate-leaf missing-body/contact warnings. The contact sensor still resolves all 30 G1 rigid bodies and the environment steps successfully. Treat this as a non-blocking Isaac Sim/Isaac Lab warning unless body count or reward behavior changes.
- `Humanoid-*` configurations import successfully, but their SMPL asset and motion payloads are not present in this repository checkout, so only the G1 task has end-to-end runtime coverage.

## Recorded installation

- Setup date: 2026-08-15
- System packages added: `python3.12-dev`, `libgl1-mesa-dev`, `libx11-dev`, `libxcursor-dev`, `libxi-dev`, `libxinerama-dev`, and `libxrandr-dev`.
- Virtual environment: `.venv`, CPython 3.12.13 (`aarch64`)
- Isaac Lab commit: `0caae64dc7c08b3fec9e748b165e4ecb211194cc` (`develop`)
- Installed versions: Isaac Sim 6.0.1.0, Isaac Lab 16.2.1, Isaac Lab Tasks 16.3.0, Isaac Lab RL 0.15.0, RSL-RL 5.4.1, PyTorch 2.11.0+cu130, torchvision 0.26.0+cu130, torchaudio 2.11.0+cu130, W&B 0.19.11, MoviePy 1.0.3, imageio-ffmpeg 0.6.0, and whole-body-tracking 0.1.0 editable.
- Verified without launching Kit: CUDA is available on NVIDIA GB10; `isaaclab`, `isaaclab_tasks`, `isaaclab_rl`, `rsl_rl`, and `wandb` import successfully; W&B login, prior run lookup, motion artifact lookup, and model checkpoint download succeed.
- G1 assets installed and `unitree_description/urdf/g1/main.urdf` verified.
- NVIDIA Omniverse EULA accepted by the user on 2026-08-15. Direct `isaacsim` import and a minimal headless Kit start/close both pass on NVIDIA GB10.
- Source migration completed on 2026-08-15. A one-environment G1 scene initializes, resets, builds policy/critic observations of 160/286 elements, applies domain randomization, and steps successfully with the verified 50 Hz legacy W&B motion.
- Full `play.py` validation loads the verified legacy `model_199.pt`, converts it in memory, exports ONNX, runs policy inference, and exits cleanly with `--max_steps 5`.
- A 16-environment, 2-iteration TensorBoard smoke training completed 768 simulation steps and saved current RSL-RL 5 checkpoints. Playback of the resulting checkpoint passed, and its ONNX model passes `onnx.checker` with seven outputs and deployment metadata.
- Bounded playback of `Tracking-Flat-G1-Low-Freq-v0` also passes at a 0.04 s control step; its reference phase uses the stored motion FPS and its render interval follows the increased decimation.
- The configured `WBT_WANDB_MODEL_PATH` was resolved through the W&B public API in an interactive zsh, downloaded, converted, exported, and played successfully with a bounded one-step run. No validation run was written to W&B.
- A public-cloud W&B smoke training completed 2 iterations and synced metrics, checkpoints, and ONNX to
  `cyijun2k/beyondmimic/49au03zb`; this is validation only and does not replace the promoted legacy checkpoint.
- Isaac Lab 3.x video recording was runtime-validated with the legacy policy: 200 frames at 1280x720 and 50 FPS
  were written to `logs/rsl_rl/temp/videos/play/play_0000.mp4`, with G1 meshes visible and debug markers disabled.
- The verified legacy motion artifact is 50 Hz and WXYZ. The prior checkpoint contains `model_state_dict`, `obs_norm_state_dict`, and `privileged_obs_norm_state_dict`; this layout remains supported by the compatibility loader.
- `uv pip check` reports Isaac Lab's intentional overrides of Isaac Sim's older Newton, MuJoCo, schema, and typing-extension pins. These overrides are declared in `.deps/IsaacLab/pyproject.toml`. It also labels the CUDA 13 SBSA cuSPARSELt wheel as a different platform even though PyTorch CUDA initialization succeeds on the GB10.
