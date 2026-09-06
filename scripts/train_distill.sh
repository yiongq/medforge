#!/usr/bin/env bash
# SFT 训练一条龙(租卡机):SFT → 合并 LoRA → v3 协议评测 → 自动关机(可选)。
# 前置:bootstrap + setup_train_env 已跑完;教材 jsonl 已传到机器;.env 已放好。
# 用法:bash scripts/train_distill.sh [配置,默认 configs/sft_distill_qwen35_4b_lora.yaml] [run 前缀,默认 distill]
#      SHUTDOWN=1 bash scripts/train_distill.sh   # 评测完直接 shutdown(AutoDL 支持实例内关机)
#
# 一切都能用环境变量覆盖(位置参数优先,不传时行为与从前逐字相同),第二阶段的弃权训练走同一个脚本:
#   CFG=configs/sft_abstain_qwen35_4b_lora.yaml RUN_PREFIX=abstain bash scripts/train_distill.sh
#   CFG=… OUTPUT_DIR=… DATASET=… RUN_PREFIX=… EVAL_ARMS=smoke,sample,abstain bash scripts/train_distill.sh
# OUTPUT_DIR / DATASET 默认从配置里读:单一事实源仍是 yaml(swift sft 只认 yaml),
# 覆盖它们只改本脚本取路径的方式(教材检查 / 找 checkpoint / merge),不一致时会告警。
set -euo pipefail
DEFAULT_CFG="configs/sft_distill_qwen35_4b_lora.yaml"
CFG="${1:-${CFG:-$DEFAULT_CFG}}"
PREFIX="${2:-${RUN_PREFIX:-distill}}"
# 换了配置却忘了 run 前缀是会静默毁数据的:评测会以 RUN_PREFIX=distill 写进
# reports/runs/distill-v3-sample/(abstain_report 唯一的参照 run,已入库),而 eval/run.py 的
# 协议闸拦不住——--served-model-name 也叫 "distill",解码/提示词/抽样卷逐字相同,
# check_protocol 判定「同一套协议」直接放行,旧答卷被 "w" 覆写、断点续跑还会复用旧模型的输出。
[ "$CFG" = "$DEFAULT_CFG" ] || [ -n "${RUN_PREFIX:-}" ] || [ -n "${2:-}" ] || {
  echo "✗ 换了配置($CFG)却没给 RUN_PREFIX:评测会覆写 reports/runs/distill-v3-*"
  echo "  例:CFG=$CFG RUN_PREFIX=abstain bash scripts/train_distill.sh"; exit 2; }
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models}"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
SWIFT="$HOME/swift-env/bin/swift"
CFG_OUT=$(grep -E "^output_dir:" "$CFG" | awk '{print $2}')
CFG_DATA=$(grep -E "^dataset:" "$CFG" | awk '{print $2}')
OUT="${OUTPUT_DIR:-$CFG_OUT}"
DATA="${DATASET:-$CFG_DATA}"
[ "$OUT" = "$CFG_OUT" ] || echo "! OUTPUT_DIR=$OUT ≠ 配置里的 $CFG_OUT:swift sft 仍写到配置那个目录"
[ "$DATA" = "$CFG_DATA" ] || echo "! DATASET=$DATA ≠ 配置里的 $CFG_DATA:swift sft 仍读配置那份教材"
# 评测臂:弃权验收的配对输入是 $PREFIX-v3-sample(默认提示词 + v3 采样)与 distill-v3-sample
# ——逐参数同协议,只差权重;弃权教材的 user 也正是这份默认提示词(build_abstain 用
# eval.run.PROMPT_CHOICE 渲染)。$PREFIX-v3-abstain 是另一条臂(贪心 8192 + --prompt abstain),
# 量的是「换个提示词能不能白拿弃权」,只能与 base-v3-abstain 这类同协议的臂比,不要混用。
ARMS="${EVAL_ARMS:-smoke,sample,forcing,abstain}"
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
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "== [1/3] 跳过训练(SKIP_TRAIN=1),直接合并已有 checkpoint =="
else
  echo "== [1/3] swift sft $CFG $(date +%T) =="
  "$SWIFT" sft "$CFG" 2>&1 | tee "logs/train-$PREFIX.log" | { grep -iE "filter|truncat|delete|liger|train_loss|eval_loss|Error|Traceback" || true; } | tail -40
  cp "logs/train-$PREFIX.log" "reports/runs/$PREFIX-v3/train-log.txt"
fi

# 2) 合并 LoRA:取 best checkpoint(load_best_model_at_end 会把最佳权重留在最后保存的目录里,
#    保险起见按 trainer_state 找 best_model_checkpoint,找不到再退回最新)
echo "== [2/3] merge $(date +%T) =="
# set -e 下 VAR=$(cmd) 里 cmd 非零会直接退出脚本——ls 的 glob 只要有一个模式没匹配就非零,所以全部兜 || true
BEST=$( { ls -d "$OUT"/v*/checkpoint-* "$OUT"/checkpoint-* 2>/dev/null || true; } | sort -V | tail -1 || true)
STATE=$( { ls "$OUT"/v*/checkpoint-*/trainer_state.json "$OUT"/checkpoint-*/trainer_state.json 2>/dev/null || true; } | sort -V | tail -1 || true)
if [ -n "$STATE" ]; then
  B=$("$HOME/swift-env/bin/python" -c "import json,sys; print(json.load(open('$STATE')).get('best_model_checkpoint') or '')" 2>/dev/null || true)
  [ -n "$B" ] && [ -d "$B" ] && BEST="$B"
fi
[ -n "$BEST" ] && [ -d "$BEST" ] || { echo "✗ 找不到 checkpoint(output_dir=$OUT)"; ls -R "$OUT" | head; exit 2; }
echo "   checkpoint: $BEST"
"$SWIFT" export --adapters "$BEST" --merge_lora true --output_dir "$OUT/merged" 2>&1 | tail -3
ls "$OUT/merged" | head -3

# 3) v3 评测:与基座、DPO 同一套臂(采样 + 强制收尾),同一批题逐题配对;再加 abstain 臂量弃权
echo "== [3/3] eval $(date +%T) =="
pkill -f "vllm serv[e]" 2>/dev/null || true; sleep 5
RUN_PREFIX="$PREFIX" bash scripts/eval_p2_arms.sh "$OUT/merged" "$ARMS"
pkill -f "vllm serv[e]" 2>/dev/null || true
echo "✓ 全部完成 $(date +%T):reports/runs/$PREFIX-v3-*/;把小文件 rsync 回本地再关机"
echo "  弃权验收(本地):uv run python -m medforge.eval.abstain_report --run $PREFIX-v3-sample --ref distill-v3-sample"
if [ "${SHUTDOWN:-0}" = "1" ]; then sync; shutdown; fi   # 不能写成 AND-list:set -e 下不传 SHUTDOWN 时脚本会以 1 退出,套在别的脚本里会被当成失败
