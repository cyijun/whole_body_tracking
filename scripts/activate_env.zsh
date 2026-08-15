if [[ ! -f .venv/bin/activate ]]; then
    print -u2 "Run this from the whole_body_tracking repository root after creating .venv."
    return 1
fi

source .venv/bin/activate

# DGX Spark / aarch64 Isaac Sim runtime requirement.
export LD_PRELOAD=/lib/aarch64-linux-gnu/libgomp.so.1

# The user accepted the NVIDIA Omniverse EULA on 2026-08-15.
export OMNI_KIT_ACCEPT_EULA=yes

# Non-secret W&B defaults. WANDB_API_KEY stays in the user's ~/.zshrc.
export WANDB_BASE_URL=https://api.wandb.ai
export WANDB_ENTITY=cyijun2k
export WANDB_USERNAME=cyijun2k
export WANDB_PROJECT=beyondmimic
if [[ -z ${WANDB_API_KEY:-} ]]; then
    print -u2 "[WARN] WANDB_API_KEY is not loaded; start an interactive zsh (or export it) before using W&B commands."
fi

# Last known-good project references. These are convenience variables for the
# existing CLI flags; update them intentionally when promoting a newer run or
# motion artifact.
export WBT_WANDB_RUN_PATH=cyijun2k/beyondmimic/lfh2rk27
export WBT_WANDB_MODEL_PATH=cyijun2k/beyondmimic/lfh2rk27/model_199.pt
export WBT_MOTION_ARTIFACT=cyijun2k/csv_to_npz/walk1_subject1_full:latest
