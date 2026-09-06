# api-dsv4flash-v3 评测汇总

协议:model=deepseek-v4-flash · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=plain · extra_body={"thinking": {"type": "enabled"}} · thinking=on · llm_judge=True · git=3277836 · 抽样=cmexam=2000,medxpertqa=1000

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| api-dsv4flash-v3 | 1000 | 47.0% | [43.9, 50.1] | 0.0% | 2.1% | 0.0% | — |
