---
license: apache-2.0
language:
  - zh
  - en
tags:
  - medical
  - dpo
  - evaluation
  - post-training
---

# MedForge · 实验产物存档

[medforge](https://github.com/yiongq/medforge) 后训练消融实验的中间产物与原始答卷。
存档的判据是**重造成本**:GPU 小时数越贵的越该留,能一条命令重跑的一律不留
(如去污染题池 182MB,20 秒可重建,故不收录)。

## dpo/ · 自采样偏好数据(约 10 GPU 小时的产物)

| 文件 | 内容 |
|---|---|
| `dpo_samples.jsonl.gz` | 3,000 道可验证医学题 × 6 次采样的**原始解法全文**(temperature 1.0, max_tokens 8192)。这是整套 DPO 数据的源头,重跑一次约 10 GPU 小时 |
| `dpo_pairs.jsonl.gz` | 判分配对后的偏好对(约 1,000 对),ms-swift DPO 格式:`{messages, rejected_response}` |
| `dpo_labels.jsonl` | 每条解法的判定标签(correct/wrong/drop)缓存。约 2 万次 LLM 仲裁的结果,复用它可省下这笔 API 开销 |

采样题源:`FreedomIntelligence/medical-o1-verifiable-problem`,已对全部评测集做字面去污染。

## eval-outputs/ · 原始答卷

每个压缩包是一个 run 目录:`*.outputs.jsonl`(模型作答全文,含 finish_reason / completion_tokens,v3 起)、
`*.scored.jsonl`(逐题判分)、`*.usability.jsonl`(逐题标签:收尾 / 声明 / 严格口径 / 退化)、`run_meta.json`(协议指纹)、`summary.md`。

**MedXpertQA 的作答全文一律不收录**(只留判分与标签):其论文附录 A 要求不以任何形式在线分享样例,而模型思考流常整段复述题目。
2026-09-04 起旧包已按此重新打包;需要复现的请按 GitHub 仓库的评测命令自行生成。

| 包 | 协议 | 内容 |
|---|---|---|
| `base.tar.gz` | v1(贪心,max_tokens 2048,全量卷) | 基座 Qwen3.5-4B,W1 底分存档;v1 会截断思考型模型,已弃用 |
| `base-v2.tar.gz` / `sft-v2` / `sft-r1-v2` / `dpo-v2` | v2(贪心,8192,固定种子抽样卷) | W2 四方案:基座 / 2024 蒸馏教材 SFT / 2025 R1 蒸馏教材 SFT / 自采样 DPO |
| `base-v3-greedy.tar.gz` | v2 协议在新引擎复跑 | 与 base-v2 配对 p=0.73:引擎无漂移;同权重两次贪心之间 20% 题目对错翻转 |
| `base-v3-sample.tar.gz` | **v3**:官方卡采样参数(1.0 / 0.95 / 20 / min_p 0 / presence 1.5)+ 32768 | 三卷收尾 100%,CMExam 严格口径 74.0%(贪心 59.6%) |
| `base-v3-forcing.tar.gz` | 贪心 8192 + 撞上限时强写 `</think>\n\n答案:` 续写 32 token | 73.5%:一条 harness 补丁拿到几乎全部收益 |
| `base-v3-abstain.tar.gz` | 贪心 8192 + 提示词允许写「答案:不确定」 | 2000 题只弃权 1 题 |
| `base-v3-greedy32k.tar.gz` | 贪心 + 32768 | 预算翻四倍只 +4pp,18~55% 的题写满仍在循环 |

P2 解码裁决的完整报告:[reports/p2-decoding-arms.md](https://github.com/yiongq/medforge/blob/main/reports/p2-decoding-arms.md)。

## 评测协议

- **v2**(W2 成绩表):temperature 0 · max_tokens 8192 · 固定种子抽样卷(CMExam 2000 / MedXpertQA 1000 / CMB-val 全量 280;CMB-val 是官方 dev 池,非榜单卷)。
  贪心解码对思考型基座会导致大面积复读未收尾,旧判分会从复读段刮出答案计分——W2 审查(2026-09-03)已订正,见仓库 README。
- **v3**(2026-09-04 起默认):Qwen3.5-4B 官方卡思考模式采样参数 + 32768 预算 + 截断守卫(未收尾 / 弃权 / 缺失分列)+ finish_reason 落盘。
- 判分 = 规则层抽取答案声明(只看 `</think>` 之后)+ LLM 仲裁兜底;验证器经 200 题校准(混合一致率 96.5%,LLM 层单独 94.1%,校准集为开放题,与实际判的选择题分布不同,待重校准);弃权计错但单独统计。

> ⚠️ 研究与工程实践产物,不构成任何医疗建议。
