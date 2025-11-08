#!/usr/bin/env bash
set -euo pipefail

# 项目根
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# 统一的 checkpoints 目录（你指定的路径）
CKPT_DIR="${CKPT_DIR:-/root/workspace/tmp/largemodel/checkpoints}"

# 数据与工作目录（按需改/可用环境变量覆盖）
DATA_DIR="${DATA_DIR:-$ROOT/dataset/sumdata/train}"
VALID_DIR="${VALID_DIR:-$ROOT/dataset/sumdata/DUC2004}"
WORK_DIR="${WORK_DIR:-$ROOT/outputs/baseline}"

# 优先用 best.pt，找不到就 last.pt
CKPT_PATH="${CKPT_PATH:-$CKPT_DIR/best.pt}"
if [ ! -f "$CKPT_PATH" ]; then
  echo "[warn] $CKPT_PATH not found, try $CKPT_DIR/last.pt"
  CKPT_PATH="$CKPT_DIR/last.pt"
fi
[ -f "$CKPT_PATH" ] || { echo "[error] no checkpoint under $CKPT_DIR"; exit 1; }

# 标题风格解码参数（可在命令行覆盖）
ARGS=(
  --mode eval
  --data_dir "$DATA_DIR"
  --valid_dir "$VALID_DIR"
  --work_dir "$WORK_DIR"
  --ckpt_dir "$CKPT_DIR"
  --ckpt_path "$CKPT_PATH"

  --decode beam
  --beam_size 6
  --length_penalty 1.0
  --min_gen_len 6
  --max_len 20
  --no_repeat_ngram_size 3
  --eos_bias 0.5

  --amp_dtype bf16
  --no_compile
)

PYTHON_BIN="${PYTHON_BIN:-python}"
mkdir -p "$WORK_DIR"
LOG="$WORK_DIR/eval.log"

echo ">>> Eval ckpt: $CKPT_PATH"
$PYTHON_BIN "$ROOT/src/main.py" \
  "${ARGS[@]}" "$@" 2>&1 | tee "$LOG"