#!/usr/bin/env bash
# 训练环境(ms-swift)独立 venv——绝不与推理栈(vllm)同锅:
# 两者的 transformers/torch 约束会互相拖后腿(实测连锁降级到不认 qwen3_5 的旧版)。
# W2 训练前在 GPU 机上执行一次;swift 命令 = ~/swift-env/bin/swift
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
unset http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY 2>/dev/null || true

SYS_PY="$(ls /root/miniconda3/bin/python3.12 2>/dev/null || command -v python3.12)"
[ -d ~/swift-env ] || uv venv ~/swift-env --python "$SYS_PY"  # 幂等:新版 uv 对已存在的 venv 会报错而非覆盖
# TORCH_BACKEND=cu129 时走 cu129 变体(驱动只到 CUDA 12.x 的主机用;默认 cu130)
UV_DEFAULT_INDEX="https://mirrors.aliyun.com/pypi/simple" \
  uv pip install --python ~/swift-env/bin/python ${TORCH_BACKEND:+--torch-backend=$TORCH_BACKEND} -U "ms-swift[swanlab]" "qwen_vl_utils>=0.0.14" torchvision flash-linear-attention liger-kernel  # Qwen3.5 硬依赖:多模态模板+线性注意力内核;liger 顺装
# uv 建的 venv 没有 pip,freeze 用 uv 出
uv pip freeze --python ~/swift-env/bin/python > "reports/train-env-freeze-$(date +%Y%m%d).txt"
echo "✓ 训练环境就绪:~/swift-env/bin/swift sft configs/sft_qwen35_4b_lora.yaml"
