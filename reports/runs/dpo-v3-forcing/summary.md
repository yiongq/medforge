# dpo-v3-forcing 评测汇总

协议:model=dpo · max_tokens=8192 · temperature=0.0 · top_p=1.0 · top_k=-1 · min_p=0.0 · presence_penalty=0.0 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=budget-forcing · extra_body=None · thinking=on · llm_judge=True · git=3277836-dirty · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| dpo-v3-forcing | 2000 | 75.4% | [73.5, 77.3] | 0.4% | 0.0% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| dpo-v3-forcing | 280 | 60.4% | [54.5, 65.9] | 0.7% | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| dpo-v3-forcing | 1000 | 26.3% | [23.7, 29.1] | 8.3% | 0.0% | 0.0% | — |
