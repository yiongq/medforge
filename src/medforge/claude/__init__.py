"""本机 Claude Code CLI 后端(判卷 / 代理标注 / 参照臂),独立成包:与 DeepSeek/OpenAI 兼容后端隔离,
代码里不含任何账号、会话、费用信息;运行产生的会话 JSON 与日志一律不进 git(见 .gitignore)。
政策线:只做判题与答题,不做蒸馏老师——Anthropic 消费者条款第 3 条禁止用输出训练模型。"""
