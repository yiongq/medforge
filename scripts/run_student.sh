#!/usr/bin/env bash
# 换学生一条龙(租卡机):基座三卷评测 → 同一份教材蒸馏 SFT → 合并 → 三卷评测 → 小文件传 HF → 可选关机。
# 前置:autodl_bootstrap + setup_train_env 已跑完;data/raw 三卷、data/processed/sft_distill_v1.jsonl、.env 已传到机器。
# 用法:bash scripts/run_student.sh <基座,如 Qwen/Qwen3.5-9B> <前缀,如 9b> <配置,如 configs/sft_distill_qwen35_9b_lora.yaml>
#      SHUTDOWN=1 bash scripts/run_student.sh …   # 全部跑完(含上传)后 shutdown
#      SKIP_BASE=1 …                              # 基座已评过,直接训
# 为什么要先传 HF 再关机:关机后我们没法远程开机,结果留在数据盘只保 15 天;小文件(判分/标签/协议)几十 MB,
# 传到 artifacts 数据集(公开)——原始答卷不传(MedXpertQA 许可禁止公开题目与答卷,且体积大),留在数据盘。
set -euo pipefail
BASE="${1:?基座模型}"; TAG="${2:?前缀}"; CFG="${3:?训练配置}"
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/root/autodl-tmp/uv-cache}"
export PATH="$HOME/.local/bin:$PATH"
t0=$(date +%s)

if [ "${SKIP_BASE:-0}" != "1" ]; then
  echo "== [A] 基座 $BASE 三卷评测(sample 臂)$(date +%T) =="
  RUN_PREFIX="base$TAG" bash scripts/eval_p2_arms.sh "$BASE" smoke,sample
fi

echo "== [B] 蒸馏 SFT $CFG → 合并 → 评测 $(date +%T) =="
CFG="$CFG" RUN_PREFIX="distill$TAG" EVAL_ARMS=smoke,sample bash scripts/train_distill.sh

echo "== [C] 小文件传 HF(fang04/medforge-artifacts · runs/$TAG-$(date +%Y%m%d))$(date +%T) =="
uv run python - "$TAG" <<'PY'
import os, sys, datetime
from medforge.env import load_env
from huggingface_hub import HfApi
load_env(); tag = sys.argv[1]
api = HfApi(token=os.environ["HF_TOKEN"])
dest = f"runs/{tag}-{datetime.date.today():%Y%m%d}"
# 只传判分/标签/协议/汇总,原始答卷(*.outputs.jsonl)与训练输出不传
api.upload_folder(folder_path="reports/runs", path_in_repo=dest, repo_id="fang04/medforge-artifacts", repo_type="dataset",
                  allow_patterns=[f"base{tag}-v3-*/*", f"distill{tag}-v3-*/*", f"smoke-*{tag}*/*"],
                  ignore_patterns=["*.outputs.jsonl"], commit_message=f"student {tag}: base + distill runs (scored/usability/meta)")
print("✓ uploaded to", dest)
PY

echo "✓ 全部完成 $(date +%T),用时 $(( ($(date +%s)-t0)/60 )) 分钟;结果 reports/runs/{base$TAG,distill$TAG}-v3-sample/"
if [ "${SHUTDOWN:-0}" = "1" ]; then sync; sleep 20; shutdown; fi
