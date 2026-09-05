"""训练侧代码。

目前只有 GRPO 的奖励插件 grpo_reward.py——它由 GPU 机上的 ms-swift(独立 venv)加载执行,
不在本机 venv 里跑训练。本机只对它做单测:插件的判分逻辑必须和评测用的验证器同源。
"""
