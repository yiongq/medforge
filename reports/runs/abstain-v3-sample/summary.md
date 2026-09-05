# abstain-v3-sample 评测汇总

协议:model=abstain · provider=openai · effort=None · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=89688ba-dirty · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| abstain-v3-sample | 2000 | 74.5% | [72.5, 76.3] | 9.2%(主动 8.9%) | 0.0% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| abstain-v3-sample | 280 | 60.4% | [54.5, 65.9] | 9.3%(主动 8.6%) | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| abstain-v3-sample | 1000 | 22.2% | [19.7, 24.9] | 10.4%(主动 9.3%) | 0.0% | 0.0% | — |
