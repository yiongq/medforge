# base-v3-sample 评测汇总

协议:model=base · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=42 · prompt=None · prompt_sha=None · mode=None · thinking=on · llm_judge=True · git=663e3e5 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample | 2000 | 75.1% | [73.2, 76.9] | 0.0% | 0.0% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample | 280 | 61.8% | [56.0, 67.3] | 0.0% | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample | 1000 | 25.6% | [23.0, 28.4] | 0.0% | 0.0% | 0.0% | — |
