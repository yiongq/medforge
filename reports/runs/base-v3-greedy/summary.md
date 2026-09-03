# base-v3-greedy 评测汇总

协议:model=base · max_tokens=8192 · temperature=0.0 · top_p=1.0 · top_k=-1 · min_p=0.0 · presence_penalty=0.0 · seed=42 · prompt=None · prompt_sha=None · mode=None · thinking=on · llm_judge=True · git=663e3e5 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-greedy | 2000 | 59.7% | [57.5, 61.8] | 0.0% | 26.9% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-greedy | 280 | 46.4% | [40.7, 52.3] | 0.0% | 29.6% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-greedy | 1000 | 10.3% | [8.6, 12.3] | 0.0% | 71.7% | 0.0% | — |
