#!/usr/bin/env bash
#SBATCH --job-name=gim_cnn_baseline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=L40
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN
#SBATCH --output=/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN/gim_cnn_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN/gim_cnn_%j.err

set -euo pipefail

CODE_DIR="${CODE_DIR:-/share/home/u23114/tj23114/packages/yaoyaping/GIM2021/CNN}"
DATASET="${DATASET:-/share/home/u23114/tj23114/data/Yaoyaping_data/GIM2021/IGS/GIM_CNN_dataset_2021_DOY091_181.nc}"
OUTPUT_DIR="${OUTPUT_DIR:-${CODE_DIR}/results/GIM_CNN_baseline_seed42}"
PYTHON_BIN="${PYTHON_BIN:-/share/apps/miniconda3/envs/tj.pytorch2.2.1/bin/python}"

if [[ ! -f "${CODE_DIR}/train_residual_cnn.py" ]]; then
  echo "Training program not found: ${CODE_DIR}/train_residual_cnn.py" >&2
  exit 1
fi

if [[ ! -f "${DATASET}" ]]; then
  echo "Dataset not found: ${DATASET}" >&2
  exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
  exit 3
fi

echo "Host: $(hostname)"
echo "Python: ${PYTHON_BIN}"
echo "Code: ${CODE_DIR}"
echo "Dataset: ${DATASET}"
echo "Output: ${OUTPUT_DIR}"

"${PYTHON_BIN}" -c \
  "import torch, netCDF4, numpy, matplotlib; assert torch.cuda.is_available(), 'CUDA is not available on this compute node'; print('PyTorch:', torch.__version__); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"

"${PYTHON_BIN}" "${CODE_DIR}/train_residual_cnn.py" \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --test-days 3 \
  --val-fraction 0.1 \
  --seed 42 \
  --epochs 100 \
  --patience 15 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --channels 32 \
  --blocks 4 \
  --num-workers "${GIM_NUM_WORKERS:-6}" \
  --device cuda \
  --amp \
  --deterministic
