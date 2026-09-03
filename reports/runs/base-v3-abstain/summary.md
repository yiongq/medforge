# base-v3-abstain 评测汇总

协议:model=base · max_tokens=8192 · temperature=0.0 · top_p=1.0 · top_k=-1 · min_p=0.0 · presence_penalty=0.0 · seed=42 · prompt=abstain · prompt_sha=ad038943 · mode=plain · thinking=on · llm_judge=True · git=27db680 · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-abstain | 2000 | 57.6% | [55.4, 59.7] | 0.1% | 31.6% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-abstain | 280 | 45.7% | [40.0, 51.6] | 0.0% | 33.6% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-abstain | 1000 | 9.9% | [8.2, 11.9] | 0.0% | 74.1% | 0.0% | — |
