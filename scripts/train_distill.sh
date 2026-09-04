#!/usr/bin/env bash
# 蒸馏 2.0 训练一条龙(租卡机):SFT → 合并 LoRA → v3 协议评测(采样臂 + 强制收尾臂)→ 自动关机(可选)。
# 前置:bootstrap + setup_train_env 已跑完;data/processed/sft_distill_v1.jsonl 已传到机器;.env 已放好。
# 用法:bash scripts/train_distill.sh [配置,默认 configs/sft_distill_qwen35_4b_lora.yaml] [run 前缀,默认 distill]
#      SHUTDOWN=1 bash scripts/train_distill.sh   # 评测完直接 shutdown(AutoDL 支持实例内关机)
set -euo pipefail
CFG="${1:-configs/sft_distill_qwen35_4b_lora.yaml}"
PREFIX="${2:-distill}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models}"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
SWIFT="$HOME/swift-env/bin/swift"
OUT=$(grep -E "^output_dir:" "$CFG" | awk '{print $2}')
DATA=$(grep -E "^dataset:" "$CFG" | awk '{print $2}')
mkdir -p logs "reports/runs/$PREFIX-v3"
[ -f "$DATA" ] || { echo "✗ 教材不存在: $DATA"; exit 2; }
echo "== 教材 $DATA: $(wc -l < "$DATA") 条 =="

# 0) 参数名核对:ms-swift 各版本参数名变过,写错会被静默忽略而不是报错。
#    4.5.2 的 `swift sft --help` 只打印 6 行(懒解析),所以改为直接读 SftArguments 的 dataclass 字段;
#    有未知键就退出——比训完才发现某个开关没生效便宜得多
"$HOME/swift-env/bin/python" - "$CFG" <<'PY' || exit 2
import dataclasses, importlib, sys, yaml
mod = importlib.import_module("swift.arguments.sft_args")
names = set()
for n in dir(mod):
    A = getattr(mod, n)
    if n.endswith("Arguments") and dataclasses.is_dataclass(A):
        for cls in A.__mro__:
            if dataclasses.is_dataclass(cls):
                names |= {f.name for f in dataclasses.fields(cls)}
cfg = yaml.safe_load(open(sys.argv[1]))
unknown = [k for k in cfg if k not in names]
print(f"配置 {len(cfg)} 个键,ms-swift 未识别: {unknown}")
sys.exit(1 if unknown else 0)
PY
"$HOME/.local/bin/uv" pip freeze --python "$HOME/swift-env/bin/python" > "reports/train-env-freeze-$(date +%Y%m%d).txt" 2>/dev/null || true

# 1) 训练(前台,日志落盘;开训 3 分钟内看 filtered/truncated 计数——这是 W2 从没兑现的验收)
echo "== [1/3] swift sft $CFG $(date +%T) =="
"$SWIFT" sft "$CFG" 2>&1 | tee "logs/train-$PREFIX.log" | grep -iE "filter|truncat|delete|liger|train_loss|eval_loss|Error|Traceback" | tail -40
cp "logs/train-$PREFIX.log" "reports/runs/$PREFIX-v3/train-log.txt"

# 2) 合并 LoRA:取 best checkpoint(load_best_model_at_end 会把最佳权重留在最后保存的目录里,
#    保险起见按 trainer_state 找 best_model_checkpoint,找不到再退回最新)
echo "== [2/3] merge $(date +%T) =="
BEST=$(ls -d "$OUT"/v*/checkpoint-* "$OUT"/checkpoint-* 2>/dev/null | sort -V | tail -1)
STATE=$(ls "$OUT"/v*/checkpoint-*/trainer_state.json "$OUT"/checkpoint-*/trainer_state.json 2>/dev/null | sort -V | tail -1)
if [ -n "$STATE" ]; then
  B=$(python3 -c "import json,sys; print(json.load(open('$STATE')).get('best_model_checkpoint') or '')" 2>/dev/null || true)
  [ -n "$B" ] && [ -d "$B" ] && BEST="$B"
fi
echo "   checkpoint: $BEST"
"$SWIFT" export --adapters "$BEST" --merge_lora true --output_dir "$OUT/merged" 2>&1 | tail -3
ls "$OUT/merged" | head -3

# 3) v3 评测:与基座、DPO 完全同一套臂(采样 + 强制收尾),同一批题逐题配对
echo "== [3/3] eval $(date +%T) =="
pkill -f "vllm serv[e]" 2>/dev/null || true; sleep 5
RUN_PREFIX="$PREFIX" bash scripts/eval_p2_arms.sh "$OUT/merged" smoke,sample,forcing
pkill -f "vllm serv[e]" 2>/dev/null || true
echo "✓ 全部完成 $(date +%T):reports/runs/$PREFIX-v3-{sample,forcing}/;把小文件 rsync 回本地再关机"
[ "${SHUTDOWN:-0}" = "1" ] && { sync; shutdown; }
