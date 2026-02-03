#!/usr/bin/env bash
set -euo pipefail

# ==========================================
# SM sweep (10..100 step 10) for BOTH image+audio
# - starts vLLM server per percent (clean lifecycle)
# - runs request_sweep_mm.py twice (image + audio)
# - outputs per-percent logs + jsonl
# - merges all jsonl at the end
# ==========================================

# -------- Config --------
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8091}"
MODEL="${MODEL:-Qwen-Omni}"                # served model name
MODEL_TAG="${MODEL_TAG:-Qwen/Qwen2.5-Omni-7B}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
VLLM_GREEN_DEVICE="${VLLM_GREEN_DEVICE:-0}"

# Paths
BASE_DIR="${BASE_DIR:-/home/konnext/Lucas/vllm-omni/evalset/v1}"
SCRIPT_DIR="${SCRIPT_DIR:-$BASE_DIR}"     # request_sweep_mm.py lives here



LOG_DIR="${LOG_DIR:-$BASE_DIR/logs}"
OUT_DIR="${OUT_DIR:-$BASE_DIR/logs}"

IMAGE_PATH="${IMAGE_PATH:-$BASE_DIR/image_text/images/coco_342952_coco342952.jpg}"
AUDIO_PATH="${AUDIO_PATH:-$BASE_DIR/audio_text/agri_samples_esc50/wav/cow__1-81269-A-3.wav}"

# Trials
N_WARMUP="${N_WARMUP:-5}"
N_TRIAL="${N_TRIAL:-20}"
MAX_TOKENS="${MAX_TOKENS:-256}"
TEMP="${TEMP:-0.0}"



PROMPT_IMAGE="${PROMPT_IMAGE:-Describe the image in one concise sentence.}"
PROMPT_AUDIO="${PROMPT_AUDIO:-Listen to the audio and describe what is happening in one concise sentence.}"

# text-only prompt（实用但不太长）
PROMPT_TEXT="${PROMPT_TEXT:-In one or two sentences, explain what a GPU does.}"




# Modalities (force text only output)
MODALITIES="${MODALITIES:-text,audio}"


# Optional: audio send mode
AUDIO_SEND_MODE="${AUDIO_SEND_MODE:-audio_url}"

# vLLM serve args
MAX_MODEL_LEN="${MAX_MODEL_LEN:-512}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"

# Timeouts
HEALTH_TIMEOUT_SEC="${HEALTH_TIMEOUT_SEC:-240}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-240}"
STOP_TIMEOUT_SEC="${STOP_TIMEOUT_SEC:-8}"
PORT_RELEASE_TIMEOUT_SEC="${PORT_RELEASE_TIMEOUT_SEC:-12}"

RUN_TAG="${RUN_TAG:-sm_sweep_$(date +%F_%H%M%S)}"

mkdir -p "$LOG_DIR" "$OUT_DIR"

echo "[INFO] RUN_TAG=$RUN_TAG"
echo "[INFO] LOG_DIR=$LOG_DIR"
echo "[INFO] OUT_DIR=$OUT_DIR"
echo "[INFO] IMAGE_PATH=$IMAGE_PATH"
echo "[INFO] AUDIO_PATH=$AUDIO_PATH"

# -------- deps --------
need_cmd() { command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] missing: $1" >&2; exit 1; }; }
need_cmd curl
need_cmd ss
need_cmd tee
need_cmd fuser
need_cmd python3
need_cmd ts || { echo "[ERROR] missing 'ts' (moreutils). Install: sudo apt-get install moreutils" >&2; exit 1; }

# -------- helpers --------
port_listening() { ss -ltnp | grep -q ":${PORT}\b"; }

wait_port_released() {
  local t0; t0="$(date +%s)"
  while port_listening; do
    local t; t="$(date +%s)"
    if (( t - t0 >= PORT_RELEASE_TIMEOUT_SEC )); then
      echo "[WARN] Port :$PORT still listening after ${PORT_RELEASE_TIMEOUT_SEC}s" >&2
      ss -ltnp | grep ":${PORT}\b" || true
      return 1
    fi
    sleep 0.1
  done
  return 0
}

assert_gpu_free() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then return 0; fi
  local used
  used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n1 | tr -d ' ')"
  echo "[INFO] GPU memory.used=${used} MiB"
  # 你按经验设阈值：比如>2000MiB就认为没清干净
  if [[ -n "$used" && "$used" -gt 2000 ]]; then
    echo "[WARN] GPU memory still high after preclean; running port kill + gpu kill again"
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 0.2
  fi
}

cleanup_vllm() {
  pkill -TERM -f "vllm serve" || true
  pkill -TERM -f "VLLM::EngineCore" || true
  pkill -TERM -f "$(which python) -c" || true

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid --format=csv,noheader \
    | awk 'NF{print $1}' | xargs -r kill -KILL || true
  fi

  fuser -k 8091/tcp >/dev/null 2>&1 || true
}




wait_health() {
  local url="http://${HOST}:${PORT}/health"
  local t0; t0="$(date +%s)"
  while true; do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "[INFO] /health OK"
      return 0
    fi
    local t; t="$(date +%s)"
    if (( t - t0 >= HEALTH_TIMEOUT_SEC )); then
      echo "[ERROR] Health timeout ${HEALTH_TIMEOUT_SEC}s: $url" >&2
      ss -ltnp | grep ":${PORT}\b" || true
      return 1
    fi
    sleep 0.5
  done
}

ready_check() {
  local url="http://${HOST}:${PORT}/v1/chat/completions"
  local t0; t0="$(date +%s)"
  local payload
  payload="$(cat <<EOF
{"model":"$MODEL","messages":[{"role":"user","content":"ping"}],"max_tokens":1,"temperature":0,"modalities":["text"]}
EOF
)"
  while true; do
    if curl -fsS -m 30 "$url" -H "Content-Type: application/json" -d "$payload" >/dev/null 2>&1; then
      echo "[INFO] Ready check OK"
      return 0
    fi
    local t; t="$(date +%s)"
    if (( t - t0 >= READY_TIMEOUT_SEC )); then
      echo "[ERROR] Ready timeout ${READY_TIMEOUT_SEC}s: $url" >&2
      ss -ltnp | grep ":${PORT}\b" || true
      return 1
    fi
    sleep 0.5
  done
}

preclean() {
  echo "[INFO] Pre-clean port :$PORT"
  fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
  sleep 0.2
  wait_port_released || true
  assert_gpu_free || true
}

# -------- server lifecycle --------
SERVER_PID=""

start_server() {
  local percent="$1"
  local log_file="$2"

  export CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"
  export VLLM_GREEN_DEVICE="$VLLM_GREEN_DEVICE"
  export VLLM_GREEN_SM_PERCENT="$percent"

  echo "[INFO] Starting vLLM percent=$percent"
  echo "[INFO] Env: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES VLLM_GREEN_DEVICE=$VLLM_GREEN_DEVICE VLLM_GREEN_SM_PERCENT=$VLLM_GREEN_SM_PERCENT"

  (
    setsid stdbuf -oL -eL \
      vllm serve "$MODEL_TAG" --omni \
        --host 0.0.0.0 --port "$PORT" \
        --served-model-name "$MODEL" \
        --trust-remote-code \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
      2>&1 | stdbuf -oL -eL ts '[%Y-%m-%d %H:%M:%S]' | tee -a "$log_file"
  ) &
  SERVER_PID=$!
  echo "[INFO] SERVER_PID=$SERVER_PID"
}

stop_server() {
  local pid="$1"
  [[ -n "${pid:-}" ]] || return 0

  echo "[INFO] Stopping server: leader pid=$pid"

  # 1) Try graceful: kill process group of leader (best effort)
  kill -TERM "-$pid" >/dev/null 2>&1 || true

  local t0; t0="$(date +%s)"
  while kill -0 "$pid" >/dev/null 2>&1; do
    local t; t="$(date +%s)"
    if (( t - t0 >= STOP_TIMEOUT_SEC )); then
      echo "[WARN] Force kill group -$pid"
      kill -KILL "-$pid" >/dev/null 2>&1 || true
      break
    fi
    sleep 0.2
  done

  # 2) Port-based cleanup (MOST RELIABLE for uvicorn/vllm)
  if ss -ltnp | grep -q ":${PORT}\b"; then
    echo "[WARN] Port :$PORT still listening; killing by port"
    fuser -k "${PORT}/tcp" >/dev/null 2>&1 || true
    sleep 0.2
  fi

  # 3) GPU compute-app cleanup (just in case something escaped port kill)
  if command -v nvidia-smi >/dev/null 2>&1; then
    # kill only processes named vllm/python that hold GPU (safer than killing all GPU users)
    while read -r gpupid pname mem; do
      if [[ "$pname" =~ vllm|python|uvicorn ]]; then
        echo "[WARN] Killing GPU compute app pid=$gpupid pname=$pname mem=$mem"
        kill -KILL "$gpupid" >/dev/null 2>&1 || true
      fi
    done < <(nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true)
  fi

  wait_port_released || true
}


cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    stop_server "$SERVER_PID" || true
    SERVER_PID=""
  fi
}
trap cleanup EXIT

# -------- sweep runner --------
P_LIST=(10 20 30 40 50 60 70 80 90 100)

REQ_IMAGE_LIST=()
REQ_AUDIO_LIST=()
LOG_LIST=()

for P in "${P_LIST[@]}"; do
  LOG_FILE="$LOG_DIR/${RUN_TAG}_pct${P}.log"
  OUT_IMG="$OUT_DIR/${RUN_TAG}_pct${P}_image.jsonl"
  OUT_AUD="$OUT_DIR/${RUN_TAG}_pct${P}_audio.jsonl"

  LOG_LIST+=("$LOG_FILE")
  REQ_IMAGE_LIST+=("$OUT_IMG")
  REQ_AUDIO_LIST+=("$OUT_AUD")

  : > "$OUT_IMG"
  : > "$OUT_AUD"

  cleanup_vllm
  preclean
  start_server "$P" "$LOG_FILE"

  wait_health
  ready_check


# request_sweep.py & request_sweep_mm.py

  # echo "[INFO] Running IMAGE sweep pct=$P -> $OUT_IMG"
  # python3 "$SCRIPT_DIR/request_sweep_mm.py" \         # request_sweep.py & request_sweep_mm.py
  #   --host "$HOST" --port "$PORT" \
  #   --model "$MODEL" --percent "$P" \
  #   --warmup "$N_WARMUP" --trials "$N_TRIAL" \
  #   --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  #   --mode image \
  #   --prompt "$PROMPT_IMAGE" \
  #   --image-path "$IMAGE_PATH" \
  #   --out-jsonl "$OUT_IMG" \
  #   --modalities "$MODALITIES"




  # echo "[INFO] Running AUDIO sweep pct=$P -> $OUT_AUD"
  # python3 "$SCRIPT_DIR/request_sweep_mm.py" \
  #   --host "$HOST" --port "$PORT" \
  #   --model "$MODEL" --percent "$P" \
  #   --warmup "$N_WARMUP" --trials "$N_TRIAL" \
  #   --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  #   --mode audio \
  #   --audio-send-mode "$AUDIO_SEND_MODE" \
  #   --prompt "$PROMPT_AUDIO" \
  #   --audio-path "$AUDIO_PATH" \
  #   --out-jsonl "$OUT_AUD" \
  #   --modalities "$MODALITIES"



  # echo "[INFO] Running IMAGE sweep pct=$P -> $OUT_IMG"
  # python3 "$SCRIPT_DIR/request_sweep.py" \
  #   --host "$HOST" --port "$PORT" \
  #   --model "$MODEL" --percent "$P" \
  #   --warmup "$N_WARMUP" --trials "$N_TRIAL" \
  #   --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  #   --modalities "$MODALITIES" \
  #   --mode image \
  #   --prompt "$PROMPT_IMAGE" \
  #   --image-path "$IMAGE_PATH" \
  #   --out-jsonl "$OUT_IMG"

  # echo "[INFO] Running AUDIO sweep pct=$P -> $OUT_AUD"
  # python3 "$SCRIPT_DIR/request_sweep.py" \
  #   --host "$HOST" --port "$PORT" \
  #   --model "$MODEL" --percent "$P" \
  #   --warmup "$N_WARMUP" --trials "$N_TRIAL" \
  #   --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
  #   --modalities "$MODALITIES" \
  #   --mode audio \
  #   --audio-send-mode "$AUDIO_SEND_MODE" \
  #   --prompt "$PROMPT_AUDIO" \
  #   --audio-path "$AUDIO_PATH" \
  #   --out-jsonl "$OUT_AUD"

  # text-only 输出文件
  OUT_TXT="$OUT_DIR/${RUN_TAG}_text_only.jsonl"
  echo "[INFO] Running TEXT-ONLY sweep pct=$P -> $OUT_TXT"

  python3 "$SCRIPT_DIR/request_sweep.py" \
    --host "$HOST" --port "$PORT" \
    --model "$MODEL" --percent "$P" \
    --warmup "$N_WARMUP" --trials "$N_TRIAL" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMP" \
    --modalities "$MODALITIES" \
    --mode text \
    --prompt "$PROMPT_TEXT" \
    --out-jsonl "$OUT_TXT"


  stop_server "$SERVER_PID" || true
  cleanup_vllm
  SERVER_PID=""

  echo "[INFO] Done pct=$P"
done

# -------- merge all outputs --------
MERGED_IMG="$OUT_DIR/${RUN_TAG}_image_all.jsonl"
MERGED_AUD="$OUT_DIR/${RUN_TAG}_audio_all.jsonl"
MERGED_ALL="$OUT_DIR/${RUN_TAG}_all.jsonl"

: > "$MERGED_IMG"
: > "$MERGED_AUD"
: > "$MERGED_ALL"

for f in "${REQ_IMAGE_LIST[@]}"; do cat "$f" >> "$MERGED_IMG"; done
for f in "${REQ_AUDIO_LIST[@]}"; do cat "$f" >> "$MERGED_AUD"; done

cat "$MERGED_IMG" "$MERGED_AUD" > "$MERGED_ALL"

echo "[OK] Sweep complete."
echo "  image merged: $MERGED_IMG (lines=$(wc -l < "$MERGED_IMG"))"
echo "  audio merged: $MERGED_AUD (lines=$(wc -l < "$MERGED_AUD"))"
echo "  all merged  : $MERGED_ALL (lines=$(wc -l < "$MERGED_ALL"))"
