# grpo-v3-abstain 评测汇总

协议:model=grpo · provider=openai · effort=None · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=42 · prompt=abstain · prompt_sha=ad038943 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=31343d1 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| grpo-v3-abstain | 2000 | 79.3% | [77.5, 81.0] | 4.7%(主动 4.5%) | 0.1% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| grpo-v3-abstain | 280 | 61.8% | [56.0, 67.3] | 8.9%(主动 8.9%) | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| grpo-v3-abstain | 1000 | 24.4% | [21.8, 27.2] | 6.1%(主动 5.9%) | 0.0% | 0.0% | — |
