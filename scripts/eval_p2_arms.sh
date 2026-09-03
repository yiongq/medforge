#!/usr/bin/env bash
# P2 解码裁决:同一基座权重、四条只换解码/提示词的对照臂,不训练。
# 目的:先量出「答不完 / 答不出」有多少是解码协议造成的,再决定训练线追准确率还是只追成本。
#   greedy    v2 口径(temperature 0 / 8192)在新引擎上复跑——同引擎对照,顺带验证与存档一致
#   greedy32k 贪心不变、预算提到 32768:把「预算不够」与「贪心退化」拆开(预测:复读只会跑得更久)
#   sample    Qwen3.5-4B 官方卡思考模式参数(temperature 1.0 / top_p 0.95 / top_k 20 / min_p 0 /
#             presence_penalty 1.5)+ 32768 预算 + 固定 seed——官方明令禁止贪心,这才是「模型本来的水平」
#   forcing   greedy + budget forcing:撞上限时接回思考流强写「</think>\n\n答案:」再续写(s1 式)
#   abstain   greedy + 允许写「答案:不确定」的提示词变体
# 用法(GPU 机,bootstrap 已跑完,.env 已放好):
#   bash scripts/eval_p2_arms.sh [模型名] [臂列表,逗号分隔]
#   bash scripts/eval_p2_arms.sh Qwen/Qwen3.5-4B smoke          # 只冒烟
set -euo pipefail
MODEL="${1:-Qwen/Qwen3.5-4B}"
ARMS="${2:-greedy,sample,forcing,abstain,greedy32k}"
PORT="${PORT:-8000}"
export MODELSCOPE_CACHE="${MODELSCOPE_CACHE:-$PWD/models}"
export PATH="$HOME/.local/bin:$PATH"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
mkdir -p logs

# vllm 默认走 HF Hub 拉模型,不认 ModelScope 缓存:先解析成本地路径(snapshot_download 幂等)
MODEL_PATH=$(uv run python -c "from modelscope import snapshot_download; print(snapshot_download('$MODEL'))")
if ! curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null; then
  echo "== 起 vLLM($MODEL_PATH)=="
  # 双重脱离:ssh 会话断了服务也不断。max-model-len 要装下 prompt + 32768 输出(MedXpertQA 题面可达 1k token)。
  # 不挂 reasoning parser:守卫靠 content 里的 </think> 判收尾,挂了 parser 思考会被剥进 reasoning_content
  (setsid nohup uv run vllm serve "$MODEL_PATH" --served-model-name base --port "$PORT" \
     --max-model-len 36864 --gpu-memory-utilization 0.92 > logs/vllm.log 2>&1 < /dev/null &)
  for _ in $(seq 1 240); do curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null && break; sleep 5; done
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { echo "vLLM 20 分钟未就绪"; tail -30 logs/vllm.log; exit 1; }
  echo "   就绪"
fi

# 抽样卷与 v2 完全一致(同 seed 同 k 才是同一批题,k 改了就不是前缀),三卷都能与存档逐题配对
E="--endpoint http://127.0.0.1:$PORT/v1 --model base --sets cmexam,cmb-val,medxpertqa --samples cmexam=2000,medxpertqa=1000 --concurrency 32"

run() {  # $1 = run-name, 其余透传给 medforge.eval.run
  local name="$1"; shift
  echo "== $name $* =="
  uv run python -m medforge.eval.run $E --run-name "$name" "$@" > "logs/$name.log" 2>&1 || { echo "✗ $name 失败"; tail -20 "logs/$name.log"; return 1; }
  grep -v "^\[normalize\]" "logs/$name.log" | tail -6
}

for arm in ${ARMS//,/ }; do
  case "$arm" in
    smoke)
      uv run python -m medforge.eval.run --endpoint "http://127.0.0.1:$PORT/v1" --model base \
        --run-name smoke --sets cmexam --limit 3 --no-llm-judge --max-tokens 2048 > logs/smoke.log 2>&1 || { tail -20 logs/smoke.log; exit 1; }
      grep -v "^\[normalize\]" logs/smoke.log | tail -4
      # 关键检查:答卷里必须有 </think>。若新版 vLLM 自动挂了 reasoning parser,思考会被剥进
      # reasoning_content,content 里没有 </think>,整个严格口径就废了(租卡笔记记过这个坑)
      uv run python - <<'PY'
import json
rows = [json.loads(l) for l in open("reports/runs/smoke/cmexam.outputs.jsonl")]
for r in rows:
    print(f"  {r['id']} finish={r['finish_reason']} tokens={r['completion_tokens']} has</think>={'</think>' in r['output']} head={r['output'][:60]!r}")
assert all("</think>" in r["output"] or r["finish_reason"] == "length" for r in rows), "content 里没有 </think>:检查 vLLM 是否挂了 reasoning parser"
PY
      ;;
    greedy)    run base-v3-greedy    --max-tokens 8192 ;;
    greedy32k) run base-v3-greedy32k --max-tokens 32768 ;;
    sample)    run base-v3-sample    --temperature 1.0 --top-p 0.95 --top-k 20 --min-p 0 --presence-penalty 1.5 --max-tokens 32768 --seed 42 ;;
    forcing)   run base-v3-forcing   --max-tokens 8192 --mode budget-forcing ;;
    abstain)   run base-v3-abstain   --max-tokens 8192 --prompt abstain ;;
    *) echo "未知臂: $arm"; exit 2 ;;
  esac
done
echo "✓ 完成:reports/runs/base-v3-*/summary.md;别忘了 git add scored/summary/run_meta 并在控制台关机"
