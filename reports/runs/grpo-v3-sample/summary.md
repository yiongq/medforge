# grpo-v3-sample 评测汇总

协议:model=grpo · provider=openai · effort=None · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=31343d1 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| grpo-v3-sample | 2000 | 78.6% | [76.7, 80.3] | 4.0%(主动 3.8%) | 0.0% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| grpo-v3-sample | 280 | 64.3% | [58.5, 69.7] | 5.0%(主动 4.6%) | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| grpo-v3-sample | 1000 | 24.7% | [22.1, 27.5] | 4.7%(主动 4.6%) | 0.1% | 0.0% | — |
