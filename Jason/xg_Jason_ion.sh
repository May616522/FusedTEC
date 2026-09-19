#!/bin/bash
#SBATCH --job-name=xg_jason_ion
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=/share/home/u23114/tj23114/packages/yaoyaping/jason2021/xg_jason_ion_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/yaoyaping/jason2021/xg_jason_ion_%j.err

# Slurm job file for xg_Jason_ion.py.
# Trains an XGBoost model for Jason ionospheric residuals using a reproducible
# random complete-calendar-day split. All valid rows are retained before
# complete calendar days are assigned to train/validation/test.
# Complete days are randomly assigned 70%/15%/15% to train/validation/test;
# every row from one day stays in exactly one split. Rows with residual < 0 are
# removed before sampling and splitting. The model uses at most 300 boosting
# rounds with early stopping, plus explicit L1 and L2 regularization.
# Local solar-time sine/cosine replace UTC HOD sine/cosine.
# If your cluster requires them, also add its #SBATCH --partition and
# #SBATCH --account lines above. Resource limits (CPU, memory, time) should be
# adjusted to the rules of your cluster.

set -euo pipefail

SCRIPT_DIR="/share/home/u23114/tj23114/packages/yaoyaping/jason2021"
PYTHON_SCRIPT="${SCRIPT_DIR}/xg_Jason_ion.py"

# This directory must directly contain the processed CSV files.
INPUT_DIR="/share/home/u23114/tj23114/data/Yaoyaping_data/jason2021"

# Each Slurm job gets an independent directory so an earlier result cannot be
# overwritten accidentally.
EXPERIMENT_NAME="full_data_random_day_residual_nonnegative_regularized_300rounds"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${SCRIPT_DIR}/xg_Jason_ion_results/${EXPERIMENT_NAME}/${RUN_ID}"

# Fixed Jason orbital altitude, in the same unit as the CSV alt column. This
# is the median of the local 2021 data (original range: 1339.02 to 1356.56).
FIXED_ALTITUDE="1345.803773"

# A smaller fresh search is sufficient for the changed split/feature setup.
# Set OPTUNA_TIMEOUT=0 to rely only on N_TRIALS.
N_TRIALS="30"
OPTUNA_TIMEOUT="7200"
N_ESTIMATORS="300"
EARLY_STOPPING_ROUNDS="30"
REG_ALPHA="0.1"
REG_LAMBDA="1.0"
MODEL_SEED="42"
SAMPLE_FRACTION="1.0"
SAMPLE_SEED="42"
SPLIT_SEED="42"
TRAIN_RATIO="0.70"
VALIDATION_RATIO="0.15"
TEST_RATIO="0.15"

# Optional parameters (commented out by default):
# --pattern: File pattern to match (default: "*.csv")
# --recursive: Search recursively in subdirectories (add flag to enable)
# --csv-engine: CSV reading engine, one of: c, python, pyarrow (default: c)
# --read-batch-size: Number of CSV files to read in batch (default: 250)
# --max-scatter-points: Max points to plot in test figures (default: 200000)
# --save-eda: Generate exploratory data analysis plots (add flag to enable)
# --sample-fraction: fraction retained inside every day; this run uses 1.0
# --sample-seed: used only when sample-fraction is smaller than 1.0
# --split-strategy random-day: randomly assign complete days (used below)
# --split-strategy time: optional chronological extrapolation comparison

# Directly use the Python executable in LiuQingyuan's Conda environment. This
# is more reliable than `conda activate` in a non-interactive Slurm job.
PYTHON_BIN="/share/home/u23114/tj23114/miniconda3/envs/LiuQingyuan/bin/python"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 2
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Conda Python is missing or not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: Input directory does not exist: ${INPUT_DIR}" >&2
    exit 2
fi

mkdir -p "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c "import matplotlib, numpy, optuna, pandas, sklearn, xgboost"; then
    echo "ERROR: The selected Python environment is missing required packages." >&2
    echo "Required: numpy pandas matplotlib scikit-learn xgboost optuna" >&2
    echo "Activate the correct module/Conda environment in xg_Jason_ion.sh." >&2
    exit 3
fi

# Prevent numerical libraries from creating more threads than Slurm allocated.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "CPU threads: ${SLURM_CPUS_PER_TASK:-1}"
echo "Optuna: ${N_TRIALS} trials, timeout ${OPTUNA_TIMEOUT}s"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Within-day sampling fraction: ${SAMPLE_FRACTION}; all valid rows retained"
echo "Random whole-day split: train/validation/test = ${TRAIN_RATIO}/${VALIDATION_RATIO}/${TEST_RATIO} (seed=${SPLIT_SEED})"
echo "Calendar-day constraint: every row from one day stays in exactly one split"
echo "Target QC: residual < 0 removed before sampling and splitting"
echo "Training: max ${N_ESTIMATORS} rounds, early stopping=${EARLY_STOPPING_ROUNDS}"
echo "Regularization: L1 reg_alpha=${REG_ALPHA}; L2 reg_lambda=${REG_LAMBDA}"
echo "Additional model feature: gim_vtec"
echo "Time features: local solar-time sine/cosine computed from UTC datetime and longitude"

COMMAND=("${PYTHON_BIN}" -u "${PYTHON_SCRIPT}" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fixed-altitude "${FIXED_ALTITUDE}" \
    --sample-fraction "${SAMPLE_FRACTION}" \
    --sample-seed "${SAMPLE_SEED}" \
    --split-strategy random-day \
    --split-seed "${SPLIT_SEED}" \
    --train-ratio "${TRAIN_RATIO}" \
    --validation-ratio "${VALIDATION_RATIO}" \
    --test-ratio "${TEST_RATIO}" \
    --n-trials "${N_TRIALS}" \
    --optuna-timeout "${OPTUNA_TIMEOUT}" \
    --n-estimators "${N_ESTIMATORS}" \
    --early-stopping-rounds "${EARLY_STOPPING_ROUNDS}" \
    --reg-alpha "${REG_ALPHA}" \
    --reg-lambda "${REG_LAMBDA}" \
    --model-seed "${MODEL_SEED}" \
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}")

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'
srun "${COMMAND[@]}"

echo "Training completed successfully."
