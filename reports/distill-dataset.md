# 蒸馏数据集报告(sft_distill_v1)

生成:2026-09-04T16:09:56+08:00 · git 151f8e6-dirty

老师:强模型带思考出题解(思考在 reasoning_content、答案在 content,两段分开落盘);user 提示词与评测共用 `eval.run.PROMPT_CHOICE/PROMPT_OPEN`,训练与评测同一份。

参数:source=cmexam-train · accept=majority · max-per-question=2 · think∈[100,4096]token · answer≤512token · 总长≤8192 · chars-per-token=1.6 · zh-ratio≥0.0 · general-ratio=0.15

## 1. 闸门漏斗

采样条数 5173 条(覆盖 2663 道题)

| 闸门 | 剔除 | 剩余 |
|---|---|---|
| 前置:id 不在题池里(样本文件比题池旧/换了 --source) | 0 | 5173 |
| ① 结构:finish_reason ≠ stop(撞上限/被中断) | 2 | 5171 |
| ① 结构:reasoning 为空(老师没开思考) | 0 | 5171 |
| ① 结构:answer 为空 | 0 | 5171 |
| ② 答案:规则层判错 | 275 | 4896 |
| ② 答案:规则层弃权(抽不出/声明弃权),按设计不走 LLM | 12 | 4884 |
| ③ 接受条件:该题未达门槛,已判对的样本一并丢弃 | 45 | 4839 |
| ④ 长度:思考过短(< --min-think-tokens) | 802 | 4037 |
| ④ 长度:思考过长(> --max-think-tokens) | 220 | 3817 |
| ④ 长度:答案段过长(> --max-answer-tokens) | 0 | 3817 |
| ④ 长度:单条总长超训练 max_length | 0 | 3817 |
| ⑤ 格式:answer 段缺「答案:」字面 | 0 | 3817 |
| ⑤ 语言:reasoning 的 CJK 占比 < --zh-ratio-min | 0 | 3817 |
| 每题上限:--max-per-question 截掉(按长度取中位保留) | 0 | 3817 |

**最终 3817 条医疗样本,覆盖 2123 道题(采样题目的 79.7%)**;混入通用回放 673 条,合计 4490 条 → `/Users/gq/Projects/GitHub/medforge/data/processed/sft_distill_v1.jsonl`

## 2. 长度分布(估算 token,换算见模块 CHARS_PER_TOKEN)

| 指标 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 思考(reasoning) | 272 | 1517 | 3727 | 4068 |
| 答案(answer) | 37 | 55 | 77 | 157 |
| 单条总长(提示词+思考+答案) | 441 | 1692 | 3903 | 4371 |

总长 max 4371 ≤ 训练 max_length 8192:训练配置的 `max_length` 必须 ≥ 这个数,否则末段的「答案:X」会被框架截掉(8 月 SFT 的坑之一)。

## 3. 按第 k 次采样的通过率

| k | 采样条数 | ② 判对 | 判对率 | 最终入选 |
|---|---|---|---|---|
| 0 | 2582 | 2443 | 94.6% | 1889 |
| 1 | 2591 | 2441 | 94.2% | 1928 |

## 4. 待办

- TODO(人工抽检,本轮未做):**「答案对但推理不成立」规则层查不出**——闸门 ② 只判 answer 段,选择题四选一蒙对也算对。需人工抽 50 条读 reasoning,统计跳步 / 编造文献 / 推理与答案矛盾的比例;若超过 10%,考虑加一道 LLM 过程审查闸门或提高 --accept 门槛。
