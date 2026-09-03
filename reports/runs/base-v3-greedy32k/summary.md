# base-v3-greedy32k 评测汇总

协议:model=base · max_tokens=32768 · temperature=0.0 · top_p=1.0 · top_k=-1 · min_p=0.0 · presence_penalty=0.0 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=plain · thinking=on · llm_judge=True · git=5bf9623 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-greedy32k | 2000 | 63.7% | [61.6, 65.8] | 0.0% | 18.1% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-greedy32k | 280 | 49.3% | [43.5, 55.1] | 0.0% | 20.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-greedy32k | 1000 | 13.9% | [11.9, 16.2] | 0.0% | 55.5% | 0.0% | — |
