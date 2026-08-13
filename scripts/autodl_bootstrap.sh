#!/usr/bin/env bash
# AutoDL 租卡环境一键初始化(镜像选 PyTorch 2.x + CUDA 12.x 基础镜像)
# 用法:bash scripts/autodl_bootstrap.sh
# 幂等:重复执行安全。注意:GPU 机上不要裸跑 `uv sync`——uv 的 exact sync
# 会卸掉本脚本第 3 步装的 GPU 栈;需要重装依赖时重跑本脚本即可。
set -euo pipefail

echo "== [1/4] 网络:卸掉学术加速代理,全部走国内源 =="
# 学术加速代理对所有流量生效且大文件必截断(实测 uv Python 包、vllm 400MB 轮子两次翻车);
# 我们的源(阿里 PyPI / ModelScope / hf-mirror)全是国内直连,不需要代理。
# 只有 git clone github 需要加速,那一步在调用方各自处理。
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "== [2/4] uv + 项目依赖 =="
# 用官方独立安装器,不依赖镜像里的 pip/conda(非交互 shell 下 conda 不激活,pip 不在 PATH)
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
# 显式用镜像 conda 自带的 Python 3.12:uv 自装 Python 的下载会被学术加速代理截断
# (实测 curl transfer closed / tar invalid,重试 52 分钟)
SYS_PY="$(ls /root/miniconda3/bin/python3.12 2>/dev/null || command -v python3.12 || true)"
if [ -n "$SYS_PY" ]; then uv sync --python "$SYS_PY"; else uv sync; fi

echo "== [3/4] 推理/评测栈(仅 vLLM+modelscope;训练栈 W2 用 setup_train_env.sh 独立装) =="
# 教训(GPU 机实测):ms-swift 与 vllm 装进同一 venv 会让解析器连锁降级——
# ms-swift 4.4.2 + vllm 0.14 + transformers 4.57,旧 transformers 不认 qwen3_5 架构。
# 评测只需要 vllm(自带配套 transformers);训练环境必须独立 venv,互不牵制。
# 版本策略:首次跑通后,用本步骤落盘的 freeze 文件把版本回填锁死(ADR 口径 5 可复现)
UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple" \
  uv pip install -U vllm modelscope
mkdir -p reports
uv pip freeze > "reports/env-freeze-$(date +%Y%m%d).txt"
echo "   依赖快照 → reports/env-freeze-$(date +%Y%m%d).txt(报告引用数字时随 git hash 一并注明)"

echo "== [4/4] 预拉基座模型(ModelScope 源,国内直连;缓存指到仓库 models/ 与 config 对齐) =="
export MODELSCOPE_CACHE="$PWD/models"
uv run python - <<'PY'
from modelscope import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B")
PY

echo "✓ 就绪。训练前 export MODELSCOPE_CACHE=\$PWD/models;启动: swift sft configs/sft_qwen35_4b_lora.yaml"
