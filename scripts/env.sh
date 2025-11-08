#!/usr/bin/env bash
set -euo pipefail

# 项目根目录（scripts 的上一级）
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# 让 Python 能找到 src/ 下的模块（扁平结构）
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

# 显卡 / PyTorch 配置（按需）
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TORCHINDUCTOR_DISABLE_CUDAGRAPHS=1
export TF32=1

# 数据与输出目录
DATA_DIR="/root/workspace/tmp/largemodel/dataset/sumdata/train"
VALID_DIR="/root/workspace/tmp/largemodel/dataset/sumdata/DUC2004"
OUT_ROOT="/root/workspace/tmp/largemodel/outputs"

# 公共训练参数（标题风格）
COMMON_ARGS=(
  --data_dir "$DATA_DIR"
  --valid_dir "$VALID_DIR"
  --vocab_size 32000
  --batch_size 128
  --grad_accum 2
  --d_model 192
  --n_heads 3
  --num_layers 3          # 建议做消融时 baseline 用 3 层更稳
  --ff_dim 768
  --dropout 0.1
  --warmup_steps 8000
  --lr 5e-4
  --epochs 1              # 消融建议 >=5
  --amp_dtype bf16
  --no_compile
  --decode beam
  --beam_size 6
  --length_penalty 1.0
  --min_gen_len 6
  --max_len 20
  --no_repeat_ngram_size 3
  --eos_bias 0.5
  --valid_limit 2000      # 快速评测，想全量就删掉
)

# 暴露一些路径给其它脚本使用
export ROOT OUT_ROOT