#!/usr/bin/env bash
# base 底分评测(ADR 口径 1:任何训练开始前,底分必须先落盘 reports/)
# 在租卡 GPU 机上执行,前置:bash scripts/autodl_bootstrap.sh 已跑完
# 用法:bash scripts/eval_base.sh [模型名,默认 Qwen/Qwen3.5-4B]
set -euo pipefail
MODEL="${1:-Qwen/Qwen3.5-4B}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models}"

echo "== [1/3] 起 vLLM 服务(后台)=="
# vllm 默认走 HF Hub 拉模型,不认 ModelScope 缓存:先解析成本地路径再喂给它
# (snapshot_download 幂等,已缓存时秒回路径)
MODEL_PATH=$(uv run python -c "from modelscope import snapshot_download; print(snapshot_download('$MODEL'))")
echo "   模型路径: $MODEL_PATH"
uv run vllm serve "$MODEL_PATH" --served-model-name target --port 8000 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT
until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
  kill -0 $VLLM_PID || { echo "vLLM 启动失败"; exit 1; }
  sleep 3
done
echo "   vLLM 就绪"

echo "== [2/3] 冒烟:每套考卷先跑 20 题确认链路 =="
uv run python -m medforge.eval.run --endpoint http://127.0.0.1:8000/v1 \
  --model target --run-name base-smoke --limit 20

echo "== [3/3] 全量三套考卷(约 9540 题)=="
uv run python -m medforge.eval.run --endpoint http://127.0.0.1:8000/v1 \
  --model target --run-name base

echo "✓ 底分落盘 reports/runs/base/summary.md —— commit 后方可开始训练"
