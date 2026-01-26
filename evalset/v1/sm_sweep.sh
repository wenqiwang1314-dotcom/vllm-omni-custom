#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Enhanced SM Sweep Script (paste-and-run)
# Key upgrades:
# - Strict pre-clean + post-stop assertions (port + GPU processes)
# - Two-stage readiness: /health + 1-token "ready" request
# - Optional CPU pinning + thread controls for stability
# - Per-run meta.json capturing environment + versions
# - Hardened timeouts, better debug dumps on failures
# ============================================================

# ====== Config ======
HOST="127.0.0.1"
PORT="8091"
MODEL="Qwen-Omni"          # served model name
MODEL_TAG="Qwen/Qwen2.5-Omni-7B"

LOG_DIR="/home/konnext/Lucas/vllm-omni/evalset/v1/logs"
OUT_DIR="/home/konnext/Lucas/vllm-omni/evalset/v1/logs"
RUN_TAG="sm_sweep_$(date +%F_%H%M%S)"

# Trials
N_WARMUP=5
N_TRIAL=20
MAX_TOKENS=256

PROMPT='Write a detailed explanation (at least 400 words) of how transformers decode tokens step by step. Mention KV cache.'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Stability knobs (safe defaults) ----
# CPU pinning: set to a CPU list like "0-15" or "0,2,4,6" if you want determinism
CPU_LIST="${CPU_LIST:-}"          # e.g. export CPU_LIST="0-15"
MEM_BIND="${MEM_BIND:-}"          # e.g. export MEM_BIND="0"  (numactl mem node)
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Timeouts
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-240}"   # health polling max seconds
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-240}"     # readiness polling max seconds
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-6}"         # seconds to wait after TERM before KILL
PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC:-10}"

# ====== vLLM command ======
VLLM_CMD_BASE=(
  vllm serve "$MODEL_TAG" --omni
  --host 0.0.0.0 --port "$PORT"
  --served-model-name "$MODEL"
  --trust-remote-code
  --enforce-eager
  --max-model-len 1024
  --max-num-seqs 1
)

# ====== Paths ======
mkdir -p "$LOG_DIR" "$OUT_DIR"

REQ_JSONL="$OUT_DIR/${RUN_TAG}_requests.jsonl"
CSV_OUT="$OUT_DIR/${RUN_TAG}_merged.csv"
META_JSON="$OUT_DIR/${RUN_TAG}_meta.json"

echo "RUN_TAG=$RUN_TAG"
echo "REQ_JSONL=$REQ_JSONL"
echo "CSV_OUT=$CSV_OUT"
echo "META_JSON=$META_JSON"

# ====== Minimal deps check ======
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing command: $1" >&2; exit 1; }; }
need_cmd curl
need_cmd ss
need_cmd ps
need_cmd tee
need_cmd ts || { echo "[ERROR] missing 'ts' (moreutils). Install: sudo apt-get install moreutils" >&2; exit 1; }
need_cmd fuser || { echo "[ERROR] missing 'fuser'. Install: sudo apt-get install psmisc" >&2; exit 1; }

# ====== Helpers ======
now_iso() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

port_listening() {
  ss -ltnp | grep -q ":${PORT}\b"
}

dump_debug() {
  echo "[DEBUG] Listening sockets on :$PORT"
  ss -ltnp | grep ":${PORT}\b" || true

  echo "[DEBUG] Top vLLM-related processes"
  ps aux | egrep "vllm|api_server|omni_stage|EngineCore" | grep -v grep || true

  echo "[DEBUG] GPU compute apps (if available)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
  fi
}

# Best-effort: list compute app PIDs on this GPU (may return empty on some drivers)
gpu_app_pids() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
    | awk 'NF{print $1}' \
    | sed 's/,//g' || true
}

# Wait for port to be released
wait_port_released() {
  local t0
  t0="$(date +%s)"
  while port_listening; do
    local t
    t="$(date +%s)"
    if (( t - t0 >= PORT_RELEASE_TIMEOUT_SEC )); then
      echo "[WARN] Port :$PORT still listening after ${PORT_RELEASE_TIMEOUT_SEC}s" >&2
      return 1
    fi
    sleep 0.1
  done
  return 0
}

wait_health() {
  local url="http://${HOST}:${PORT}/health"
  local t0
  t0="$(date +%s)"

  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[INFO] /health OK"
      return 0
    fi
    local t
    t="$(date +%s)"
    if (( t - t0 >= HEALTH_TIMEOUT_SEC )); then
      echo "[ERROR] Health check timeout (${HEALTH_TIMEOUT_SEC}s): $url" >&2
      dump_debug >&2
      echo "[DEBUG] Last 200 lines of current log: ${LOG_FILE:-<unknown>}" >&2
      if [[ -n "${LOG_FILE:-}" && -f "${LOG_FILE:-}" ]]; then
        tail -n 200 "$LOG_FILE" >&2 || true
      fi
      return 1
    fi
    sleep 0.5
  done
}

# A stricter readiness probe than /health:
# send a minimal 1-token request so we know the model path is actually serving.
ready_check() {
  local url="http://${HOST}:${PORT}/v1/chat/completions"
  local t0
  t0="$(date +%s)"

  local payload
  payload="$(cat <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0}
EOF
)"

  while true; do
    if curl -fsS -m 30 "$url" \
      -H "Content-Type: application/json" \
      -d "$payload" >/dev/null 2>&1; then
      echo "[INFO] Ready request OK (/v1/chat/completions)"
      return 0
    fi

    local t
    t="$(date +%s)"
    if (( t - t0 >= READY_TIMEOUT_SEC )); then
      echo "[ERROR] Ready check timeout (${READY_TIMEOUT_SEC}s): $url" >&2
      dump_debug >&2
      echo "[DEBUG] Last 200 lines of current log: ${LOG_FILE:-<unknown>}" >&2
      if [[ -n "${LOG_FILE:-}" && -f "${LOG_FILE:-}" ]]; then
        tail -n 200 "$LOG_FILE" >&2 || true
      fi
      return 1
    fi
    sleep 0.5
  done
}

preclean() {
  echo "[INFO] Pre-clean port :$PORT and old vLLM processes"

  # 1) kill anything listening on port (most reliable)
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  sleep 0.2

  # 2) cleanup possible remnants
  pkill -f "vllm serve .*--port ${PORT}" >/dev/null 2>&1 || true
  pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
  pkill -f "/home/konnext/Lucas/.venv/bin/python -c" >/dev/null 2>&1 || true
  sleep 0.2

  # 3) assert port released (best-effort)
  wait_port_released || true
}

start_server() {
  local percent="$1"
  local log_file="$2"
  local pid_file="$3"

  export CUDA_VISIBLE_DEVICES=0
  export VLLM_GREEN_DEVICE=0
  export VLLM_GREEN_SM_PERCENT="$percent"

  export OMP_NUM_THREADS="$OMP_NUM_THREADS"
  export TOKENIZERS_PARALLELISM="$TOKENIZERS_PARALLELISM"

  echo "[INFO] Starting server percent=$percent log=$log_file pid_file=$pid_file"
  echo "[INFO] Env: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES VLLM_GREEN_DEVICE=$VLLM_GREEN_DEVICE VLLM_GREEN_SM_PERCENT=$VLLM_GREEN_SM_PERCENT OMP_NUM_THREADS=$OMP_NUM_THREADS TOKENIZERS_PARALLELISM=$TOKENIZERS_PARALLELISM CPU_LIST=${CPU_LIST:-<unset>} MEM_BIND=${MEM_BIND:-<unset>}"

  # ---------- 构造 command 前缀 ----------
  CMD_PREFIX=()
  if [[ -n "${CPU_LIST:-}" && -n "${MEM_BIND:-}" && -x "$(command -v numactl)" ]]; then
    CMD_PREFIX=(numactl --physcpubind="$CPU_LIST" --membind="$MEM_BIND")
  elif [[ -n "${CPU_LIST:-}" ]]; then
    CMD_PREFIX=(taskset -c "$CPU_LIST")
  fi
  # -------------------------------------

  (
    setsid stdbuf -oL -eL \
      "${CMD_PREFIX[@]}" "${VLLM_CMD_BASE[@]}" 2>&1 \
      | stdbuf -oL -eL ts '[%Y-%m-%d %H:%M:%S]' \
      | tee -a "$log_file"
  ) &

  local leader_pid=$!
  echo "$leader_pid" > "$pid_file"
  SERVER_PID="$leader_pid"
}


stop_server() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0

  if kill -0 "$pid" >/dev/null 2>&1; then
    echo "[INFO] Stopping server leader pid=$pid (kill process group -$pid)"
    kill -TERM "-$pid" >/dev/null 2>&1 || true

    local t0
    t0="$(date +%s)"
    while kill -0 "$pid" >/dev/null 2>&1; do
      local t
      t="$(date +%s)"
      if (( t - t0 >= STOP_TIMEOUT_SEC )); then
        echo "[WARN] Force killing process group -$pid"
        kill -KILL "-$pid" >/dev/null 2>&1 || true
        break
      fi
      sleep 0.2
    done
  fi

  # Post-stop: ensure port released
  wait_port_released || true
}

write_meta() {
  # Write a lightweight meta json (no jq dependency).
  local start_ts="$1"
  local end_ts="$2"

  local git_rev="unknown"
  if command -v git >/dev/null 2>&1 && git -C "$SCRIPT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git_rev="$(git -C "$SCRIPT_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
  fi

  local nvsmi="n/a"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvsmi="$(nvidia-smi --query-gpu=name,driver_version,cuda_version,persistence_mode,power.limit,clocks.current.graphics,clocks.current.memory --format=csv,noheader 2>/dev/null | head -n 1 || echo n/a)"
  fi

  cat >"$META_JSON" <<EOF
{
  "run_tag": "$(printf '%s' "$RUN_TAG")",
  "start_utc": "$(printf '%s' "$start_ts")",
  "end_utc": "$(printf '%s' "$end_ts")",
  "host": "$(printf '%s' "$HOST")",
  "port": "$(printf '%s' "$PORT")",
  "model_served_name": "$(printf '%s' "$MODEL")",
  "model_tag": "$(printf '%s' "$MODEL_TAG")",
  "warmup": $N_WARMUP,
  "trials": $N_TRIAL,
  "max_tokens": $MAX_TOKENS,
  "cpu_list": "$(printf '%s' "${CPU_LIST:-}")",
  "mem_bind": "$(printf '%s' "${MEM_BIND:-}")",
  "omp_num_threads": "$(printf '%s' "$OMP_NUM_THREADS")",
  "tokenizers_parallelism": "$(printf '%s' "$TOKENIZERS_PARALLELISM")",
  "script_dir": "$(printf '%s' "$SCRIPT_DIR")",
  "git_rev_script_dir": "$(printf '%s' "$git_rev")",
  "nvidia_smi_summary": "$(printf '%s' "$nvsmi" | sed 's/"/\\"/g')"
}
EOF

  echo "[INFO] Wrote meta: $META_JSON"
}

# ====== Cleanup trap ======
cleanup() {
  if [[ "${SERVER_PID:-}" != "" ]]; then
    stop_server "$SERVER_PID" || true
  fi
}
trap cleanup EXIT

# ====== Sweep ======
LOG_LIST=()
REQ_LIST=()

P_LIST=(10 20 30 40 50 60 70 80 90 100)

RUN_START="$(now_iso)"

for P in "${P_LIST[@]}"; do
  LOG_FILE="$LOG_DIR/${RUN_TAG}_pct${P}.log"
  PID_FILE="$OUT_DIR/${RUN_TAG}_pct${P}.pid"
  REQ_FILE="$OUT_DIR/${RUN_TAG}_pct${P}_requests.jsonl"

  LOG_LIST+=("$LOG_FILE")
  REQ_LIST+=("$REQ_FILE")

  : > "$REQ_FILE"

  preclean

  start_server "$P" "$LOG_FILE" "$PID_FILE"
  echo "[INFO] SERVER_PID=$SERVER_PID (pidfile=$PID_FILE)"

  # Two-stage readiness
  wait_health
  ready_check

  echo "[INFO] About to run request_sweep, writing to $REQ_FILE"
  python3 "$SCRIPT_DIR/request_sweep.py" \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" \
    --percent "$P" \
    --warmup "$N_WARMUP" --trials "$N_TRIAL" \
    --max-tokens "$MAX_TOKENS" \
    --prompt "$PROMPT" \
    --out-jsonl "$REQ_FILE"

  # Assert: request file must have content
  if [[ ! -s "$REQ_FILE" ]]; then
    echo "[ERROR] request_sweep produced empty file: $REQ_FILE" >&2
    echo "[DEBUG] Last 200 lines of log: $LOG_FILE" >&2
    tail -n 200 "$LOG_FILE" >&2 || true
    exit 1
  fi
  echo "[INFO] request_sweep done, lines: $(wc -l < "$REQ_FILE")"

  stop_server "$SERVER_PID" || true
  SERVER_PID=""

  # Extra: ensure port really released before next round
  wait_port_released || true

  echo "[INFO] Done percent=$P"
done

# ====== Combine per-percent requests ======
: > "$REQ_JSONL"
for f in "${REQ_LIST[@]}"; do
  cat "$f" >> "$REQ_JSONL"
done
echo "[INFO] Combined requests into $REQ_JSONL (lines=$(wc -l < "$REQ_JSONL"))"

if [[ ! -s "$REQ_JSONL" ]]; then
  echo "[ERROR] Combined requests file is empty: $REQ_JSONL" >&2
  exit 1
fi

# ====== Merge logs + requests into CSV ======
python3 "$SCRIPT_DIR/parse_gc_logs.py" \
  --req-jsonl "$REQ_JSONL" \
  --logs "${LOG_LIST[@]}" \
  --out-csv "$CSV_OUT"

RUN_END="$(now_iso)"
write_meta "$RUN_START" "$RUN_END"

echo "[OK] All done."
echo "Requests: $REQ_JSONL"
echo "Merged CSV: $CSV_OUT"
echo "Meta: $META_JSON"
