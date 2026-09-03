# base-v3-forcing 评测汇总

协议:model=base · max_tokens=8192 · temperature=0.0 · top_p=1.0 · top_k=-1 · min_p=0.0 · presence_penalty=0.0 · seed=42 · prompt=default · prompt_sha=1f256db8 · mode=budget-forcing · thinking=on · llm_judge=True · git=27db680 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-forcing | 2000 | 74.5% | [72.5, 76.3] | 0.1% | 0.0% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-forcing | 280 | 59.6% | [53.8, 65.2] | 0.4% | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-forcing | 1000 | 27.5% | [24.8, 30.3] | 0.2% | 0.0% | 0.0% | — |
