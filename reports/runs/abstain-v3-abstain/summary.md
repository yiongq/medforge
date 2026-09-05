# abstain-v3-abstain 评测汇总

协议:model=abstain · provider=openai · effort=None · max_tokens=8192 · temperature=0.0 · top_p=1.0 · top_k=-1 · min_p=0.0 · presence_penalty=0.0 · seed=42 · prompt=abstain · prompt_sha=ad038943 · mode=plain · extra_body=None · thinking=on · llm_judge=True · git=89688ba-dirty · 抽样=cmexam=2000,medxpertqa=1000

## cmexam

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| abstain-v3-abstain | 2000 | 75.9% | [74.0, 77.8] | 4.4%(主动 4.4%) | 4.3% | 0.0% | — |

## cmb-val

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| abstain-v3-abstain | 280 | 60.0% | [54.2, 65.6] | 7.5%(主动 7.5%) | 6.8% | 0.0% | — |

## medxpertqa

| 配置 | n | 准确率 | 95% CI | 弃权率 | 未收尾率 | 缺失率 | vs base |
|---|---|---|---|---|---|---|---|
| abstain-v3-abstain | 1000 | 23.3% | [20.8, 26.0] | 1.7%(主动 1.7%) | 13.3% | 0.0% | — |
