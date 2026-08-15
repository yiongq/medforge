#!/usr/bin/env bash
# 部署与压测一条龙(在租用 GPU 机上执行):
#   BF16 起服务 → 并发扫描 → 关;FP8 起服务 → 并发扫描 → 关 → 汇总报告
# 用法:bash scripts/serve_bench.sh [模型路径或HF名] [卡型标签]
#
# 为什么比 BF16 与 FP8 而不是 AWQ:两者用同一份权重、无需额外量化步骤,
# 变量干净(只差数值精度);AWQ 要先跑校准量化,属另一条实验线。
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
MODEL="${1:-fang04/medforge-qwen3.5-4b-dpo}"
GPU="${2:-unknown}"
cd "$(dirname "$0")/.."
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true

serve_and_bench () {   # $1=label  $2...=vllm 附加参数
  local label="$1"; shift
  echo "== [$label] 起 vLLM =="
  # 注:--disable-log-requests 在新版 vLLM 已移除,别加
  uv run vllm serve "$MODEL" --served-model-name target --port 8000 \
    "$@" > "vllm_$label.log" 2>&1 &
  local pid=$!
  until curl -sf http://127.0.0.1:8000/v1/models >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo "vLLM 启动失败($label)"; tail -30 "vllm_$label.log"; return 1; }
    sleep 3
  done
  echo "   就绪,开始压测"
  uv run python -m medforge.serve.bench --endpoint http://127.0.0.1:8000/v1 \
    --model target --label "$label" --gpu "$GPU"
  kill $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
  sleep 8   # 等显存彻底释放,否则下一档起服务会 OOM
}

serve_and_bench bf16
serve_and_bench fp8 --quantization fp8   # 权重动态转 FP8,无需校准数据

uv run python -m medforge.serve.bench_report
echo "✓ 压测完成:reports/deployment.md + reports/bench-*.json"
