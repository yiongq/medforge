# base-v3-sample-s44 评测汇总

协议:model=base · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=44 · prompt=default · prompt_sha=1f256db8 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=3277836-dirty · 抽样=cmexam=2000,medxpertqa=1000

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample-s44 | 280 | 61.1% | [55.2, 66.6] | 2.5% | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample-s44 | 1000 | 27.1% | [24.4, 29.9] | 4.3% | 0.0% | 0.0% | — |
