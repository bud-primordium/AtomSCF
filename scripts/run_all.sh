#!/bin/bash
# 主脚本：运行所有原子计算
# 按顺序执行：H → C → 额外原子（He, Li, Al）
# 日志文件：../logs/run_all.log

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

LOG_FILE="../logs/run_all.log"
mkdir -p ../logs

echo "=============================================" | tee "$LOG_FILE"
echo "AtomSCF 作业计算全流程" | tee -a "$LOG_FILE"
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

START_TIME=$(date +%s)

# 1. 氢原子
echo "" | tee -a "$LOG_FILE"
echo "═══ 阶段 1/3：氢原子计算 ═══" | tee -a "$LOG_FILE"
bash run_h_calculations.sh
echo "✓✓✓ 氢原子计算完成" | tee -a "$LOG_FILE"

# 2. 碳原子
echo "" | tee -a "$LOG_FILE"
echo "═══ 阶段 2/3：碳原子计算 ═══" | tee -a "$LOG_FILE"
bash run_c_calculations.sh
echo "✓✓✓ 碳原子计算完成" | tee -a "$LOG_FILE"

# 3. 额外原子
echo "" | tee -a "$LOG_FILE"
echo "═══ 阶段 3/3：额外原子计算 ═══" | tee -a "$LOG_FILE"
bash run_extra_atoms.sh
echo "✓✓✓ 额外原子计算完成" | tee -a "$LOG_FILE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# 总结
echo "" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"
echo "✅ 所有计算全部完成！" | tee -a "$LOG_FILE"
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "总用时: $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒" | tee -a "$LOG_FILE"
echo "=============================================" | tee -a "$LOG_FILE"

# 统计生成的文件
echo "" | tee -a "$LOG_FILE"
echo "生成的 JSON 文件（共 $(ls ../test_results/*.json 2>/dev/null | wc -l | xargs) 个）：" | tee -a "$LOG_FILE"
ls -lh ../test_results/*.json 2>/dev/null | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "日志文件位置：" | tee -a "$LOG_FILE"
echo "  - 总日志：logs/run_all.log" | tee -a "$LOG_FILE"
echo "  - 氢原子：logs/h_calculations.log" | tee -a "$LOG_FILE"
echo "  - 碳原子：logs/c_calculations.log" | tee -a "$LOG_FILE"
echo "  - 额外原子：logs/extra_atoms.log" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "下一步：运行 plot_comparisons.py 生成波函数对比图" | tee -a "$LOG_FILE"
