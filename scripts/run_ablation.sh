#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "${HERE}/env.sh"

PY=${PYTHON:-python}

run_exp () {
  local name="$1"; shift
  local outdir="${OUT_ROOT}/${name}"
  mkdir -p "$outdir"
  echo ">>> Running $name -> $outdir"
  # 直接调用文件（扁平 src/）
  $PY -u "${ROOT}/src/main.py" \
    --work_dir "$outdir" \
    "${COMMON_ARGS[@]}" \
    "$@" 2>&1 | tee "${outdir}/train.log"
}

# 1) baseline
run_exp baseline

# 2) 去掉位置编码
run_exp ablate_no_posenc --ablate no_posenc

# 3) 单头注意力
run_exp ablate_single_head --ablate single_head

# 4) 去掉 label smoothing
run_exp ablate_no_label_smoothing --ablate no_label_smoothing