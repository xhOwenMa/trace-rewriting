#!/usr/bin/env bash
# Full anti-distillation (AD) experiment on GSM8K using our optimized rewrite
# instruction for the rewriter LLM. Runs the complete paper pipeline end to
# end on the full GSM8K train (generation/distillation) and test (evaluation)
# splits.
#
# All heavy artifacts (HF hub cache, vLLM cache, dataset cache, generated
# traces, distilled student weights, vLLM logs) live in scratch; only
# lightweight results (eval metrics, trace dataset, sample report, config)
# are copied back into the project folder at the end.
#
# Stages:
#   1. Serve teacher with vLLM, generate original clean traces with
#      `generation_instruction`, kill server.
#   2. Serve rewriter with vLLM, rewrite traces with our optimized
#      `rewrite_instruction` + answer-force, kill server.
#   3. Distill a student on the rewritten traces (LoRA SFT).
#   4. Evaluate the distilled student on the full GSM8K test split.
#   5. Copy non-weight results to <project>/results/gsm8k/<run>/.
#
# Stages 1+2 are automatically skipped when pre-generated rewritten traces
# are present at data/<dataset>/optimized/ (shipped with the repo). To force
# regeneration, delete or rename that directory.
#
# Expected env vars (see README.md):
#   HUGGING_FACE_HUB_TOKEN  [required] HF token with access to the gated
#                           teacher and student models.
#   NUM_GPUS                [required-ish, default 2] Number of GPUs on the
#                           machine. The default rewriter (gpt-oss-120b)
#                           requires at least 2x A100 80GB / 1x H100. Set to
#                           1 only if you swap in a smaller rewriter.
#   SCRATCH_BASE            [optional, default /tmp/trace-rewriting-cache]
#                           Parent dir for HF / torch / vLLM caches, traces,
#                           student weights, and logs. Can easily exceed
#                           100GB; point at a large filesystem if /tmp is
#                           limited.
#
# Fully optional overrides: RUN, STUDENT_MODEL, PROJECT_DIR, VLLM_PORT,
# VLLM_READY_TIMEOUT, VLLM_EXTRA_ARGS.

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR:-"$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"}
cd "${PROJECT_DIR}"

SCRATCH_BASE=${SCRATCH_BASE:-/tmp/trace-rewriting-cache}
NUM_GPUS=${NUM_GPUS:-2}

DATASET=gsm8k
RUN=${RUN:-ad_optimized_inst_gsm8k}
STUDENT_MODEL=${STUDENT_MODEL:-meta-llama/Llama-3.2-3B}

VLLM_PORT=${VLLM_PORT:-8000}
VLLM_READY_TIMEOUT=${VLLM_READY_TIMEOUT:-1800}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

# ---------- preflight ----------
if [[ -z "${HUGGING_FACE_HUB_TOKEN:-}${HF_TOKEN:-}" ]]; then
    echo "error: HUGGING_FACE_HUB_TOKEN is not set." >&2
    echo "  The teacher (${TEACHER:-DeepSeek-R1-Distill-Qwen-7B}) and student" >&2
    echo "  (${STUDENT_MODEL}) models are gated on the Hugging Face Hub." >&2
    echo "  Create a token at https://huggingface.co/settings/tokens, then:" >&2
    echo "      export HUGGING_FACE_HUB_TOKEN=hf_..." >&2
    exit 1
fi

# ---------- scratch paths ----------
SCRATCH_ROOT="${SCRATCH_BASE}/trace-rewriting"
SCRATCH_WORKDIR="${SCRATCH_ROOT}/runs"
SCRATCH_RUN="${SCRATCH_WORKDIR}/${DATASET}/${RUN}"
STUDENT_OUT="${SCRATCH_RUN}/student"
EVAL_OUT="${SCRATCH_RUN}/eval"
LOG_DIR="${SCRATCH_RUN}/logs"

mkdir -p "${SCRATCH_RUN}" "${LOG_DIR}"

# ---------- pre-generated data detection ----------
# If the repo already ships rewritten traces for this dataset under
# data/<dataset>/optimized/, reuse them and skip the (expensive) teacher
# generation and rewriter rewriting stages. Detection looks for the HF
# `dataset_info.json` marker so a stray empty directory won't false-trigger.
PRE_GENERATED_DATA="${PROJECT_DIR}/data/${DATASET}/optimized"
if [[ -f "${PRE_GENERATED_DATA}/dataset_info.json" ]]; then
    USE_PREGEN=1
    TRACES="${PRE_GENERATED_DATA}"
else
    USE_PREGEN=0
    TRACES="${SCRATCH_RUN}/traces"
fi

# ---------- cache redirection (HF, torch, triton, vLLM, tmp) ----------
export HF_HOME="${SCRATCH_ROOT}/hf"
export HF_HUB_CACHE="${HF_HOME}/hub"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export HF_MODULES_CACHE="${HF_HOME}/modules"
export HF_METRICS_CACHE="${HF_HOME}/metrics"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export TORCH_HOME="${SCRATCH_ROOT}/torch"
export TRITON_CACHE_DIR="${SCRATCH_ROOT}/triton"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg_cache"
export PIP_CACHE_DIR="${SCRATCH_ROOT}/pip"
export VLLM_CACHE_ROOT="${SCRATCH_ROOT}/vllm"
export TMPDIR="${SCRATCH_ROOT}/tmp"
mkdir -p "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}" "${HF_MODULES_CACHE}" \
         "${HF_METRICS_CACHE}" "${TRANSFORMERS_CACHE}" "${TORCH_HOME}" \
         "${TRITON_CACHE_DIR}" "${XDG_CACHE_HOME}" "${PIP_CACHE_DIR}" \
         "${VLLM_CACHE_ROOT}" "${TMPDIR}"

export OPENAI_BASE_URL="http://localhost:${VLLM_PORT}/v1"
export OPENAI_API_KEY=${OPENAI_API_KEY:-EMPTY}

# ---------- read teacher/rewriter from config ----------
TEACHER=$(python - <<'PY'
import yaml; print(yaml.safe_load(open("config/generate.yaml"))["teacher_model_name"])
PY
)
REWRITER=$(python - <<'PY'
import yaml; print(yaml.safe_load(open("config/generate.yaml"))["rewriter_model_name"])
PY
)

# ---------- vLLM lifecycle helpers ----------
VLLM_PID=""

stop_vllm() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" 2>/dev/null; then
        echo "[vllm] stopping pid ${VLLM_PID}..."
        kill "${VLLM_PID}" 2>/dev/null || true
        for _ in $(seq 1 30); do
            kill -0 "${VLLM_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[vllm] SIGTERM ignored, sending SIGKILL"
            kill -9 "${VLLM_PID}" 2>/dev/null || true
        fi
        wait "${VLLM_PID}" 2>/dev/null || true
        sleep 3  # let the driver release GPU memory before the next stage
    fi
    VLLM_PID=""
}

trap 'stop_vllm' EXIT INT TERM

start_vllm() {
    local model=$1
    local log_file="${LOG_DIR}/vllm_$(echo "${model}" | tr '/:' '__').log"
    echo "[vllm] starting ${model} on port ${VLLM_PORT} (log: ${log_file})"
    # shellcheck disable=SC2086
    vllm serve "${model}" --port "${VLLM_PORT}" \
        --tensor-parallel-size "${NUM_GPUS}" \
        --download-dir "${HF_HUB_CACHE}" ${VLLM_EXTRA_ARGS} \
        > "${log_file}" 2>&1 &
    VLLM_PID=$!

    local elapsed=0
    while (( elapsed < VLLM_READY_TIMEOUT )); do
        if curl -sf "${OPENAI_BASE_URL}/models" > /dev/null 2>&1; then
            echo "[vllm] ready (pid ${VLLM_PID}) after ${elapsed}s"
            return 0
        fi
        if ! kill -0 "${VLLM_PID}" 2>/dev/null; then
            echo "[vllm] process died during startup. Tail of ${log_file}:"
            tail -n 40 "${log_file}" >&2 || true
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
    done
    echo "[vllm] not ready within ${VLLM_READY_TIMEOUT}s. Tail of ${log_file}:"
    tail -n 40 "${log_file}" >&2 || true
    exit 1
}

echo "=========================================================="
echo "Dataset        : ${DATASET} (full train + test splits)"
echo "Teacher        : ${TEACHER}"
echo "Rewriter       : ${REWRITER}"
echo "Student        : ${STUDENT_MODEL}"
echo "Num GPUs       : ${NUM_GPUS}"
echo "Scratch run    : ${SCRATCH_RUN}"
echo "HF_HOME        : ${HF_HOME}"
echo "VLLM_CACHE_ROOT: ${VLLM_CACHE_ROOT}"
echo "vLLM endpoint  : ${OPENAI_BASE_URL}"
echo "Traces         : ${TRACES}$( ((USE_PREGEN)) && echo ' (pre-generated, stages 1+2 will be skipped)')"
echo "=========================================================="

if (( USE_PREGEN )); then
    echo "[1/5] Skipped — using pre-generated traces from ${TRACES}"
    echo "[2/5] Skipped — pre-generated traces already include rewrite_trace + answer-forcing"
else
    echo "[1/5] Serving teacher and generating original clean traces"
    start_vllm "${TEACHER}"
    python src/generate.py --config config/generate.yaml \
        --dataset "${DATASET}" \
        --working_dir "${SCRATCH_WORKDIR}" \
        --experiment_folder "${RUN}" \
        --skip_rewrite --skip_af
    stop_vllm

    echo "[2/5] Serving rewriter and rewriting traces + answer-forcing"
    start_vllm "${REWRITER}"
    python src/generate.py --config config/generate.yaml \
        --dataset "${DATASET}" \
        --working_dir "${SCRATCH_WORKDIR}" \
        --experiment_folder "${RUN}" \
        --original_traces_path "${TRACES}"
    stop_vllm
fi

echo "[3/5] Distilling student ${STUDENT_MODEL} on rewritten traces"
accelerate launch --num_processes "${NUM_GPUS}" src/distill.py \
    --config config/distill.yaml \
    --student_model_name "${STUDENT_MODEL}" \
    --data_dir "${TRACES}" \
    --trace_colname rewrite_trace \
    --lora \
    --output_dir "${STUDENT_OUT}"

echo "[4/5] Evaluating distilled student on ${DATASET} test"
python src/evaluate.py \
    --model_name_or_path "${STUDENT_OUT}/final_model" \
    --dataset "${DATASET}" \
    --tensor_parallel_size "${NUM_GPUS}" \
    --out_dir "${EVAL_OUT}"

echo "[5/5] Copying lightweight results back to project (no model weights)"
PROJECT_RESULTS="${PROJECT_DIR}/results/${DATASET}/${RUN}"
mkdir -p "${PROJECT_RESULTS}"
# Excludes:
#   student/    — distilled model weights + LoRA adapter + train checkpoints
#   logs/       — vLLM server logs
#   *.safetensors, *.bin, *.pt — any stray weight files
rsync -a \
    --exclude='student/' \
    --exclude='logs/' \
    --exclude='*.safetensors' \
    --exclude='*.bin' \
    --exclude='*.pt' \
    "${SCRATCH_RUN}/" "${PROJECT_RESULTS}/"

echo
echo "Pipeline complete."
echo "  scratch run : ${SCRATCH_RUN}"
echo "  student wts : ${STUDENT_OUT}/final_model   (kept on scratch)"
echo "  project copy: ${PROJECT_RESULTS}            (no weights)"
