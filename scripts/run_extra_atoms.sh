#!/bin/bash
# 额外原子计算脚本：He, Li, Al 的 LSDA 计算
# 输出目录：../test_results/
# 日志文件：../logs/extra_atoms.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

LOG_FILE="logs/extra_atoms.log"
mkdir -p logs test_results

echo "=======================================" | tee "$LOG_FILE"
echo "额外原子 LSDA 计算（He, Li, Al）" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"

# 原子列表
declare -a ATOMS=(2 3 13)
declare -a NAMES=("He" "Li" "Al")

for i in "${!ATOMS[@]}"; do
  Z="${ATOMS[$i]}"
  NAME="${NAMES[$i]}"

  echo "" | tee -a "$LOG_FILE"
  echo "[$((i+1))/3] 运行 $NAME 原子（Z=$Z）LSDA-VWN 计算..." | tee -a "$LOG_FILE"

  python AtomSCF/examples/run_atom.py \
    --Z "$Z" \
    --method DFT \
    --mode LSDA \
    --xc VWN \
    --n 2000 \
    --eigs-per-l 3 \
    --export "test_results/${NAME}_lsda.json" \
    --compare-ref 2>&1 | tee -a "$LOG_FILE"

  echo "✓ $NAME 完成" | tee -a "$LOG_FILE"
done

# 总结
echo "" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo "额外原子计算全部完成" | tee -a "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=======================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "生成的文件：" | tee -a "$LOG_FILE"
ls -lh test_results/He_*.json test_results/Li_*.json test_results/Al_*.json 2>/dev/null | tee -a "$LOG_FILE"
