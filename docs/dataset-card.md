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

## eval-outputs/ · 四方案的完整原始答卷

`base`(协议 v1)、`base-v2`、`sft-v2`(2024 蒸馏教材)、`sft-r1-v2`(2025 R1 蒸馏教材),
每份含三套考卷的 `*.outputs.jsonl`(模型作答全文)与 `*.scored.jsonl`(逐题判分)。

有一个细节值得一看:同样是三套考卷的全部答卷,**基座压缩后 26 MB,而抄过旧教材的模型只有 1.3 MB**
——蒸馏 SFT 对思考长度的压制,连文件体积都藏不住。

## 评测协议(v2)

temperature 0 · max_tokens 8192 · 固定种子抽样卷(CMExam 2000 / MedXpertQA 1000 / CMB-val 全量)。
判分 = 规则层抽取答案声明 + LLM 仲裁兜底;验证器经 200 题人工校准(一致率 96.5%);
弃权计错但单独统计。完整报告见 [GitHub 仓库](https://github.com/yiongq/medforge)。

> ⚠️ 研究与工程实践产物,不构成任何医疗建议。
