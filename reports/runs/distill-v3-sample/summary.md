# distill-v3-sample 评测汇总

协议:model=distill · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=3277836-dirty · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| distill-v3-sample | 2000 | 80.4% | [78.6, 82.1] | 0.2% | 0.0% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| distill-v3-sample | 280 | 65.7% | [60.0, 71.0] | 0.7% | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| distill-v3-sample | 1000 | 26.4% | [23.8, 29.2] | 0.7% | 0.0% | 0.2% | — |
