#!/usr/bin/env bash
# base 底分评测入口(ADR 口径 1:任何训练开始前,本脚本产出的底分必须先落盘 reports/)
# W1b 实装内容:
#   1) vllm serve 起 base 模型(OpenAI 兼容端口)
#   2) EvalScope 自定义 benchmark 跑 cmexam-test / cmb-val / medxpertqa
#      —— 注意:四套考卷都不在 EvalScope 内置列表,判分 adapter 必须复用 medforge.verify(口径唯一)
#   3) medforge.eval.report 汇总 → reports/base.md
set -euo pipefail
echo "TODO(W1b): 见文件头注释;当前为占位,防止 W2 训练先于底分开跑" >&2
exit 2
