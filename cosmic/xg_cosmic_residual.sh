#!/bin/bash
#SBATCH --job-name=xg_cosmic_residual
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --chdir=/share/home/u23114/tj23114/packages/yaoyaping/cosmic2021
#SBATCH --output=/share/home/u23114/tj23114/packages/yaoyaping/cosmic2021/xg_cosmic_residual_%j.out
#SBATCH --error=/share/home/u23114/tj23114/packages/yaoyaping/cosmic2021/xg_cosmic_residual_%j.err

# COSMIC residual 的 XGBoost Slurm 作业脚本。
# 保留 10% 完整 DOY 作为独立测试集，其余日期执行 3 折 GroupKFold。
# Optuna 自动搜索 XGBoost 参数，每折最多 500 轮并使用 early stopping。
# Kp/Dst 滞后特征由逐小时 OMNI 文件生成，不使用未来数据。

set -euo pipefail

# 程序与数据位于不同目录，分别使用明确的绝对路径。
SCRIPT_DIR="/share/home/u23114/tj23114/packages/yaoyaping/cosmic2021"
PYTHON_SCRIPT="${SCRIPT_DIR}/xg_cosmic_residual.py"

# 如环境不同，仍可在提交作业时用同名环境变量覆盖下面三个路径。
PYTHON_BIN="${PYTHON_BIN:-/share/home/u23114/tj23114/miniconda3/envs/LiuQingyuan/bin/python}"
INPUT_DIR="${INPUT_DIR:-/share/home/u23114/tj23114/data/Yaoyaping_data/cosmic2021}"
OMNI_FILE="${OMNI_FILE:-/share/home/u23114/tj23114/data/Yaoyaping_data/Other/omni2_2021.dat.csv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/xg_cosmic_results}"

EXPERIMENT_NAME="complete_doy_groupkfold3_optuna_lag_500rounds"
RUN_ID="${SLURM_JOB_ID:-local_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXPERIMENT_NAME}/${RUN_ID}"

N_ESTIMATORS="500"
EARLY_STOPPING_ROUNDS="30"
CV_FOLDS="3"
OPTUNA_TRIALS="30"
OPTUNA_TIMEOUT_MINUTES="0"
OPTUNA_SEED="42"
TAIL_QUANTILE="0.10"
TAIL_WEIGHT="2.0"
LEARNING_RATE="0.05"
MAX_DEPTH="6"
MIN_CHILD_WEIGHT="5"
SUBSAMPLE="0.80"
COLSAMPLE_BYTREE="0.80"
REG_ALPHA="0.10"
REG_LAMBDA="1.0"
SPLIT_SEED="42"
MODEL_SEED="42"
SAMPLING_SEED="42"

# 0 表示每个年积日保留全部有效行。试跑时可设为例如 5000；即使抽样，
# 也是在每个 DOY 内独立随机抽取，之后仍以完整 DOY 为单位划分集合。
MAX_ROWS_PER_DAY="0"

if [[ ! -f "${PYTHON_SCRIPT}" ]]; then
    echo "ERROR: Python script not found: ${PYTHON_SCRIPT}" >&2
    exit 2
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable is unavailable: ${PYTHON_BIN}" >&2
    exit 2
fi
if [[ ! -d "${INPUT_DIR}" ]]; then
    echo "ERROR: input directory does not exist: ${INPUT_DIR}" >&2
    exit 2
fi
if [[ ! -f "${OMNI_FILE}" ]]; then
    echo "ERROR: OMNI CSV is missing: ${OMNI_FILE}" >&2
    exit 2
fi
if [[ ! -f "${INPUT_DIR}/cosmic2_model_2021_001.csv" ]]; then
    echo "ERROR: first daily CSV is missing: ${INPUT_DIR}/cosmic2_model_2021_001.csv" >&2
    exit 2
fi
if [[ ! -f "${INPUT_DIR}/cosmic2_model_2021_365.csv" ]]; then
    echo "ERROR: last daily CSV is missing: ${INPUT_DIR}/cosmic2_model_2021_365.csv" >&2
    exit 2
fi

shopt -s nullglob
CSV_FILES=("${INPUT_DIR}"/cosmic2_model_2021_???.csv)
if [[ "${#CSV_FILES[@]}" -ne 365 ]]; then
    echo "ERROR: expected 365 daily CSV files, found ${#CSV_FILES[@]} in ${INPUT_DIR}" >&2
    exit 2
fi
shopt -u nullglob

mkdir -p "${OUTPUT_DIR}"

if ! "${PYTHON_BIN}" -c "import matplotlib,numpy,optuna,pandas,sklearn,xgboost"; then
    echo "ERROR: missing Python packages: numpy pandas matplotlib scikit-learn xgboost optuna" >&2
    exit 3
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Python script: ${PYTHON_SCRIPT}"
echo "Python executable: ${PYTHON_BIN}"
echo "Input: ${INPUT_DIR}"
echo "OMNI: ${OMNI_FILE}"
echo "Output: ${OUTPUT_DIR}"
echo "Daily CSV files: ${#CSV_FILES[@]}"
echo "Split: isolated 10% complete-DOY test; remaining DOYs use ${CV_FOLDS}-fold GroupKFold"
echo "Optuna: trials=${OPTUNA_TRIALS}, timeout_minutes=${OPTUNA_TIMEOUT_MINUTES}, seed=${OPTUNA_SEED}"
echo "Target: residual (excluded from model features to prevent leakage)"
echo "Boosting: max ${N_ESTIMATORS}, early stopping ${EARLY_STOPPING_ROUNDS}"
echo "Tail weighting: lower/upper ${TAIL_QUANTILE} quantiles, weight=${TAIL_WEIGHT}"
echo "Regularization: L1=${REG_ALPHA}, L2=${REG_LAMBDA}"
echo "Sampling: max rows per DOY=${MAX_ROWS_PER_DAY}; 0 means all rows"

COMMAND=("${PYTHON_BIN}" -u "${PYTHON_SCRIPT}"
    --input-dir "${INPUT_DIR}"
    --omni-file "${OMNI_FILE}"
    --output-dir "${OUTPUT_DIR}"
    --year 2021
    --train-ratio 0.70
    --validation-ratio 0.20
    --test-ratio 0.10
    --cv-folds "${CV_FOLDS}"
    --optuna-trials "${OPTUNA_TRIALS}"
    --optuna-timeout-minutes "${OPTUNA_TIMEOUT_MINUTES}"
    --optuna-seed "${OPTUNA_SEED}"
    --split-seed "${SPLIT_SEED}"
    --model-seed "${MODEL_SEED}"
    --sampling-seed "${SAMPLING_SEED}"
    --max-rows-per-day "${MAX_ROWS_PER_DAY}"
    --n-estimators "${N_ESTIMATORS}"
    --early-stopping-rounds "${EARLY_STOPPING_ROUNDS}"
    --tail-quantile "${TAIL_QUANTILE}"
    --tail-weight "${TAIL_WEIGHT}"
    --learning-rate "${LEARNING_RATE}"
    --max-depth "${MAX_DEPTH}"
    --min-child-weight "${MIN_CHILD_WEIGHT}"
    --subsample "${SUBSAMPLE}"
    --colsample-bytree "${COLSAMPLE_BYTREE}"
    --reg-alpha "${REG_ALPHA}"
    --reg-lambda "${REG_LAMBDA}"
    --n-jobs "${SLURM_CPUS_PER_TASK:-1}"
    --device cpu)

printf 'Command:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

if command -v srun >/dev/null 2>&1 && [[ -n "${SLURM_JOB_ID:-}" ]]; then
    srun "${COMMAND[@]}"
else
    "${COMMAND[@]}"
fi

echo "Training completed successfully."
