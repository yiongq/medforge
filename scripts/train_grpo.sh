#!/usr/bin/env bash
# 第三阶段 GRPO 一条龙(租卡机):前置体检 → GRPO 训练 → 合并 LoRA → v3 协议双提示词臂评测 → 可选关机。
# 前置:bootstrap + setup_train_env 已跑完;data/processed/grpo_{train,eval}.jsonl 已传到机器;.env 已放好。
# 用法:bash scripts/train_grpo.sh [配置,默认 configs/grpo_qwen35_4b_lora.yaml] [run 前缀,默认 grpo]
#      SKIP_TRAIN=1 bash scripts/train_grpo.sh          # 跳过训练,直接合并已有 checkpoint 再评
#      SHUTDOWN=1   bash scripts/train_grpo.sh          # 评测完直接 shutdown
#      VLLM_VERSION=0.27.1 bash scripts/train_grpo.sh   # 指定要装进训练 venv 的 vllm 版本
# 与 train_distill.sh 的差别只有两处:多了第 0.5 步的 vllm 前置(colocate rollout 要在训练进程内起 vLLM),
# 评测多跑一条弃权提示词臂(第三阶段训的就是「该不该弃权」,只看默认提示词看不出东西)。
set -euo pipefail
CFG="${1:-configs/grpo_qwen35_4b_lora.yaml}"
PREFIX="${2:-grpo}"
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"; export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models}"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
SWIFT="$HOME/swift-env/bin/swift"
SWIFT_PY="$HOME/swift-env/bin/python"
UV="$HOME/.local/bin/uv"
OUT=$(grep -E "^output_dir:" "$CFG" | awk '{print $2}')
DATA=$(grep -E "^dataset:" "$CFG" | awk '{print $2}')
VAL=$(grep -E "^val_dataset:" "$CFG" | awk '{print $2}' || true)
mkdir -p logs "reports/runs/$PREFIX-v3"
[ -f "$DATA" ] || { echo "✗ 题单不存在: $DATA(先跑 medforge.data.build_grpo)"; exit 2; }
echo "== 训练题单 $DATA: $(wc -l < "$DATA") 条 =="
if [ -n "${VAL:-}" ]; then
  [ -f "$VAL" ] || { echo "✗ 验证题单不存在: $VAL(build_grpo --eval-n 200)"; exit 2; }
  echo "== 验证题单 $VAL: $(wc -l < "$VAL") 条 =="
fi

# 0) 参数名核对:与 train_distill.sh 同一道闸,只是换成 RLHFArguments(GRPO 的键散在
#    RLHFArguments / GRPOArguments / GRPOArgumentsMixin / RolloutTrainerArgumentsMixin / VllmArguments
#    五个 dataclass 里,走 __mro__ 一次收全)。写错的键会被静默忽略而不是报错——训完才发现最贵。
"$SWIFT_PY" - "$CFG" <<'PY' || exit 2
import dataclasses, importlib, sys, yaml
mod = importlib.import_module("swift.arguments.rlhf_args")
A = getattr(mod, "RLHFArguments")
names = set()
for cls in A.__mro__:
    if dataclasses.is_dataclass(cls):
        names |= {f.name for f in dataclasses.fields(cls)}
cfg = yaml.safe_load(open(sys.argv[1]))
unknown = [k for k in cfg if k not in names]
print(f"配置 {len(cfg)} 个键,ms-swift 未识别: {unknown}")
sys.exit(1 if unknown else 0)
PY

# 0.5) vllm 前置:GRPO 的 colocate rollout 是在**训练进程内**起 vLLM 的(vllm_mode: colocate),
#      而 setup_train_env.sh 刻意没把 vllm 装进 ~/swift-env——推理栈与训练栈分 venv 是硬规矩
#      (docs/gpu-租卡实操笔记.md「环境安装的四条军规」第 3 条:ms-swift + vllm 同锅会让解析器连锁降级,
#       实测降到不认 qwen3_5 架构的旧 transformers)。这里是那条规矩唯一必须破例的地方,所以:
#      装之前打印版本、装之后再打印一次,并显式验证 transformers 仍认得 qwen3_5;不认就当场中止,
#      因为此时训练能起来但模型架构解析已经坏了,错误会推迟到加载权重时才炸(甚至更晚)。
versions() {  # $1 = 说明文字。装 vllm 前后各打一次,回看日志时才知道解析器把谁动了
  "$SWIFT_PY" -c '
import sys
out = []
for m in ["torch", "transformers", "trl", "peft", "swift", "vllm"]:
    try:
        out.append(f"{m}={__import__(m).__version__}")
    except Exception as e:
        out.append(f"{m}=<{type(e).__name__}>")
print("   [" + sys.argv[1] + "] " + " ".join(out))
' "$1"
}
echo "== [0.5/4] 训练 venv 的 vllm 前置 $(date +%T) =="
versions "before"
if "$SWIFT_PY" -c "import vllm" 2>/dev/null; then
  echo "   vllm 已在训练 venv 里,跳过安装"
else
  # 版本对齐推理 venv:那边是跑通过的组合,不要在这里另挑一个版本再赌一次
  VLLM_VERSION="${VLLM_VERSION:-$("$UV" run python -c "import vllm; print(vllm.__version__)" 2>/dev/null || true)}"
  [ -n "$VLLM_VERSION" ] || { echo "✗ 推理 venv 里也没有 vllm,先跑 scripts/autodl_bootstrap.sh,或显式传 VLLM_VERSION=x.y.z"; exit 2; }
  echo "   装 vllm==$VLLM_VERSION 进 ~/swift-env(仅此一次;失败不要改成 uv sync,那会连 ms-swift 一起重解)"
  UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple" \
    "$UV" pip install --python "$SWIFT_PY" ${TORCH_BACKEND:+--torch-backend=$TORCH_BACKEND} "vllm==$VLLM_VERSION" || {
      echo "✗ vllm 装不进训练 venv;驱动只到 CUDA 12.x 时按笔记加 --extra-index-url https://wheels.vllm.ai/$VLLM_VERSION/cu129 TORCH_BACKEND=cu129"; exit 2; }
  versions "after"
  # 硬闸:transformers 被 vllm 拖降到不认 qwen3_5 就地中止。比版本号比较可靠——
  # 真正要的是「架构解析得动」,而不是某个版本区间(笔记里翻车那次版本号看着也正常)
  "$SWIFT_PY" - <<'PY' || { echo "✗ transformers 被降级到不认 qwen3_5:回滚 venv(重跑 scripts/setup_train_env.sh)再议"; exit 2; }
import sys
import transformers
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
ok = "qwen3_5" in CONFIG_MAPPING_NAMES
print(f"transformers={transformers.__version__} 认识 qwen3_5: {ok}")
sys.exit(0 if ok else 1)
PY
  "$SWIFT_PY" -c "import swift, trl; print('swift', swift.__version__, 'trl', trl.__version__)" \
    || { echo "✗ 装完 vllm 后 ms-swift/trl 反而 import 不动了:回滚 venv"; exit 2; }
fi
"$UV" pip freeze --python "$SWIFT_PY" > "reports/train-env-freeze-$(date +%Y%m%d).txt" 2>/dev/null || true

# 1) 训练。前 50 步必看 completions 采样(log_completions: true):beta=0 不带参考模型,
#    跑飞的表现是格式先崩(不再写「答案:X」)而 reward 看着还在动
if [ "${SKIP_TRAIN:-0}" = "1" ]; then
  echo "== [1/4] 跳过训练(SKIP_TRAIN=1),直接合并已有 checkpoint =="
else
  echo "== [1/4] swift rlhf $CFG $(date +%T) =="
  "$SWIFT" rlhf "$CFG" 2>&1 | tee "logs/train-$PREFIX.log" \
    | { grep -iE "reward|completion|Error|Traceback|OutOfMemory|vllm|sleep" || true; } | tail -60
  cp "logs/train-$PREFIX.log" "reports/runs/$PREFIX-v3/train-log.txt"
fi

# 2) 合并 LoRA。GRPO 没有 load_best_model_at_end(eval 出的是奖励不是 loss),取最新 checkpoint;
#    仍按 train_distill.sh 的写法兜住 glob 非零退出(set -e 下 VAR=$(cmd) 会直接把脚本带走)
echo "== [2/4] merge $(date +%T) =="
BEST=$( { ls -d "$OUT"/v*/checkpoint-* "$OUT"/checkpoint-* 2>/dev/null || true; } | sort -V | tail -1 || true)
[ -n "$BEST" ] && [ -d "$BEST" ] || { echo "✗ 找不到 checkpoint(output_dir=$OUT)"; ls -R "$OUT" 2>/dev/null | head; exit 2; }
echo "   checkpoint: $BEST"
"$SWIFT" export --adapters "$BEST" --merge_lora true --output_dir "$OUT/merged" 2>&1 | tail -3
ls "$OUT/merged" | head -3

# 3) 评测臂 A:默认提示词。起 vLLM + 冒烟 + v3 采样臂,run 名 = $PREFIX-v3-sample
echo "== [3/4] eval(默认提示词)$(date +%T) =="
pkill -f "vllm serv[e]" 2>/dev/null || true; sleep 5
RUN_PREFIX="$PREFIX" bash scripts/eval_p2_arms.sh "$OUT/merged" smoke,sample

# 4) 评测臂 B:弃权提示词。刻意不用 eval_p2_arms.sh 自带的 abstain 臂——那条是贪心口径(v2 遗留),
#    与 sample 臂差的就不止「提示词」一个变量了。第三阶段要回答的问题是「同一套解码下,
#    允许说不确定时模型的弃权是否变得聪明」,所以采样参数逐个抄 sample 臂,只改 --prompt。
#    vLLM 仍是上一步起的那个(served-model-name = $PREFIX),不重启。
echo "== [4/4] eval(弃权提示词)$(date +%T) =="
PORT="${PORT:-8000}"
SETS="${SETS:-cmexam,cmb-val,medxpertqa}"
uv run python -m medforge.eval.run \
  --endpoint "http://127.0.0.1:$PORT/v1" --model "$PREFIX" --sets "$SETS" \
  --samples cmexam=2000,medxpertqa=1000 --concurrency 32 \
  --run-name "$PREFIX-v3-abstain" --prompt abstain \
  --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 \
  --max-tokens 32768 --seed 42 > "logs/$PREFIX-v3-abstain.log" 2>&1 \
  || { echo "✗ 弃权臂失败"; tail -20 "logs/$PREFIX-v3-abstain.log"; exit 1; }
grep -v "^\[normalize\]" "logs/$PREFIX-v3-abstain.log" | tail -6

pkill -f "vllm serv[e]" 2>/dev/null || true
echo "✓ 全部完成 $(date +%T):reports/runs/$PREFIX-v3-{sample,abstain}/;把小文件 rsync 回本地再关机"
[ "${SHUTDOWN:-0}" = "1" ] && { sync; shutdown; }
