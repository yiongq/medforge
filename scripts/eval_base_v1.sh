#!/usr/bin/env bash
# 【存档脚本】协议 v1 的 base 底分评测(max_tokens 2048、贪心、全量卷),reports/runs/base/ 由它产出。
# v1 会截断思考型模型,已被 v2/v3 取代;保留只为复现存档,不要用它跑新评测——
# 新评测用 scripts/eval_p2_arms.sh(v3 协议)或直接 uv run python -m medforge.eval.run(默认即 v3)。
# 用法:bash scripts/eval_base_v1.sh [模型名,默认 Qwen/Qwen3.5-4B]
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
  --model target --run-name base --adopt-legacy --max-tokens 2048 --temperature 0 --top-p 1 --top-k -1 --presence-penalty 0-v1-smoke --limit 20 --max-tokens 2048 --temperature 0 --top-p 1 --top-k -1 --presence-penalty 0

echo "== [3/3] 全量三套考卷(约 9540 题)=="
uv run python -m medforge.eval.run --endpoint http://127.0.0.1:8000/v1 \
  --model target --run-name base --adopt-legacy --max-tokens 2048 --temperature 0 --top-p 1 --top-k -1 --presence-penalty 0

echo "✓ 底分落盘 reports/runs/base/summary.md —— commit 后方可开始训练"
