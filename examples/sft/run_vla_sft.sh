#! /bin/bash
#SBATCH -p accelerated #accelerated
#SBATCH --gres=gpu:1     # 1 GPUs
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=6
#SBATCH -J train_pi0.5
#SBATCH -o logs/train_ppo_realrobot/%x_%j.out
#SBATCH -e logs/train_ppo_realrobot/%x_%j.err

set -eo pipefail

#activate environment and set paths
SCRIPT_DIR="$( cd "$(dirname "${BASH_SOURCE[0]}")" && pwd )"
if [ -n "${SLURM_SUBMIT_DIR}" ] && [ -f "${SLURM_SUBMIT_DIR}/examples/sft/train_vla_sft.py" ]; then
    export REPO_PATH="${SLURM_SUBMIT_DIR}"
    export EMBODIED_PATH="${REPO_PATH}/examples/sft"
else
    export EMBODIED_PATH="${SCRIPT_DIR}"
    export REPO_PATH=$(dirname "$(dirname "${EMBODIED_PATH}")")
fi

module load FFmpeg/7.1.2-GCCcore-14.3.0
source "${REPO_PATH}/.venv/bin/activate"
HYDRA_FULL_ERROR=1
export SRC_FILE="${EMBODIED_PATH}/train_vla_sft.py"

export HF_HOME="${REPO_PATH}/.cache/huggingface"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "${HF_DATASETS_CACHE}" "${HUGGINGFACE_HUB_CACHE}"



export MUJOCO_GL="egl"
export PYOPENGL_PLATFORM="egl"
export USE_TF=0
export TRANSFORMERS_NO_TF=1

export PYTHONPATH=${REPO_PATH}:${LIBERO_REPO_PATH}:$PYTHONPATH

if [ -z "$1" ]; then
    CONFIG_NAME="franka_joint_sft_openpi_pi05"
else
    CONFIG_NAME=$1
fi

echo "Using Python at $(which python)"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    python "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "runner.logger.log_path=${LOG_DIR}"
)
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | tee -a "${MEGA_LOG_FILE}"
