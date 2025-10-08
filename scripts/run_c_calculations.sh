#!/bin/bash
# 碳原子计算脚本：LSDA + RHF + UHF 三方法对比
# 输出目录：../test_results/
# 日志文件：../logs/c_calculations.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_FILE="logs/c_calculations.log"
mkdir -p logs test_results

echo "=======================================" | tee "$LOG_FILE"
echo "碳原子（Z=6）计算" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"

# 1. LSDA 计算
echo "" | tee -a "$LOG_FILE"
echo "[1/3] 运行 C 原子 LSDA-VWN 计算..." | tee -a "$LOG_FILE"
python AtomSCF/examples/run_atom.py \
  --Z 6 \
  --method DFT \
  --mode LSDA \
  --xc VWN \
  --n 2000 \
  --eigs-per-l 3 \
  --export test_results/C_lsda.json \
  --compare-ref 2>&1 | tee -a "$LOG_FILE"

echo "✓ LSDA 完成" | tee -a "$LOG_FILE"

# 2. RHF 计算
echo "" | tee -a "$LOG_FILE"
echo "[2/3] 运行 C 原子 RHF 计算..." | tee -a "$LOG_FILE"
python AtomSCF/examples/run_atom.py \
  --Z 6 \
  --method HF \
  --hf-mode RHF \
  --n 2000 \
  --eigs-per-l 3 \
  --export test_results/C_rhf.json 2>&1 | tee -a "$LOG_FILE"

echo "✓ RHF 完成" | tee -a "$LOG_FILE"

# 3. UHF 计算
echo "" | tee -a "$LOG_FILE"
echo "[3/3] 运行 C 原子 UHF 计算..." | tee -a "$LOG_FILE"
python AtomSCF/examples/run_atom.py \
  --Z 6 \
  --method HF \
  --hf-mode UHF \
  --n 2000 \
  --eigs-per-l 3 \
  --export test_results/C_uhf.json 2>&1 | tee -a "$LOG_FILE"

echo "✓ UHF 完成" | tee -a "$LOG_FILE"

# 总结
echo "" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo "碳原子计算全部完成" | tee -a "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "生成的文件：" | tee -a "$LOG_FILE"
ls -lh test_results/C_*.json | tee -a "$LOG_FILE"
