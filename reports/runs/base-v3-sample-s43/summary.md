# base-v3-sample-s43 评测汇总

协议:model=base · max_tokens=32768 · temperature=1.0 · top_p=0.95 · top_k=20 · min_p=0.0 · presence_penalty=1.5 · seed=43 · prompt=default · prompt_sha=1f256db8 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=3277836-dirty · 抽样=cmexam=2000,medxpertqa=1000

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample-s43 | 280 | 58.9% | [53.1, 64.5] | 5.4% | 0.0% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| base-v3-sample-s43 | 1000 | 26.9% | [24.2, 29.7] | 3.1% | 0.0% | 0.0% | — |
