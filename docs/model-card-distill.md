---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
language:
  - zh
  - en
tags:
  - medical
  - distillation
  - sft
  - qwen3.5
  - post-training
library_name: transformers
pipeline_tag: text-generation
---

# MedForge · Qwen3.5-4B-Distill(合格老师 + 领域内题 + 修好的尺子)

在 Qwen3.5-4B 上做医疗推理后训练的第二代产物。教材由一个在**本项目考卷上被实测过**的老师(DeepSeek V4-flash,CMExam 严格口径 93.2%)
在去污染后的 CMExam 官方训练集上写出,经五道闸门筛选(收尾 / 规则层判对 / 多数一致 / 长度 / 格式)后 LoRA SFT,
训练与评测共用同一份提示词。**同一个 4B、4490 条教材、单卡 1h18m。**

> ⚠️ 研究与工程实践产物,**不构成任何医疗建议**;依据中国相关规定,AI 不得用于自动诊断与处方。

## 成绩(协议 v3:官方采样参数 · 32768 预算 · 截断守卫 · 严格口径 = 写完 ∧ 有结论 ∧ 答对;同一批题逐题配对)

| 模型 | CMExam n=2000(域内) | CMB-val n=280(迁移) | MedXpertQA n=1000(迁移,英文) | 每题 token 均值 |
|---|---|---|---|---|
| Qwen3.5-4B 基座 | 74.0% | 60.0% | 25.1% | 5.2k / 5.7k / 8.4k |
| + DPO(自采样,一代产物) | 74.7% | 58.9% | 26.1% | ≈ 基座 |
| **+ 蒸馏 2.0(本模型)** | **80.4%**(+6.5 [+4.7, +8.2],p<10⁻⁴,Holm/BH ✓) | **65.7%**(+5.7 [+0.6, +10.8],p=0.04) | 26.4%(+1.3,持平) | **1.8k / 2.3k / 4.7k** |
| DeepSeek V4-flash(老师,参照) | 93.2% | 84.3% | 47.0% | 0.6k / 1.3k / 5.4k |

三卷收尾率 100%(基座在旧的贪心协议下 26~65% 交不了卷),贪心解码下也不复读(78.7 / 69.3 / 23.8)。
完整报告:[reports/distill-2.0.md](https://github.com/yiongq/medforge/blob/main/reports/distill-2.0.md);
配对区间、多重比较校正与最小可检出差异:[reports/usability-v3-train.md](https://github.com/yiongq/medforge/blob/main/reports/usability-v3-train.md)。

**怎么读**:CMExam 的题源来自其官方训练集,该卷是「域内」成绩;泛化证据是 CMB-val(同向,但 280 题功效不足)。
MedXpertQA 老师自己只有 47%、基座错题里 64% 老师也错,可教空间小,持平符合预期。

## 与一代(8 月)蒸馏 SFT 失败的差别

8 月两版蒸馏 SFT 在旧协议下全线降分。审查(2026-09-03)证明那不是「蒸馏对思考型基座有害」,而是五处同时错:
老师未在本卷验证、题源不在域内、训练提示词与评测不同、教材 20% 被截断、评测用官方禁止的贪心解码并从复读段刮分。
本模型逐项修正:老师实测 93%、题源 CMExam-train 三通道去污染(剔 53 题)、提示词同源、长度数据侧硬筛、协议 v3。
细节见 [reports/p2-decoding-arms.md](https://github.com/yiongq/medforge/blob/main/reports/p2-decoding-arms.md)。

## 训练配置

- 基座 `Qwen/Qwen3.5-4B`,LoRA 全层(r=64, α=128, dropout 0.05),lr 5e-5 cosine,2 epoch,有效 batch 16,max_length 8192(truncation_strategy delete),
  liger kernel,按验证集 loss 选最优 checkpoint(eval_loss 1.019 → 0.963,best = 最后一步)
- 教材 `sft_distill_v1.jsonl`:3817 条医疗样本(覆盖 2123 题)+ 673 条通用回放;思考段是完整的 `<think>…</think>`,
  assistant 末行「答案:X」与评测格式一致;漏斗见 [reports/distill-dataset.md](https://github.com/yiongq/medforge/blob/main/reports/distill-dataset.md)
- 框架 ms-swift 4.5.2 · RTX 5090 单卡 · 配置文件 `configs/sft_distill_qwen35_4b_lora.yaml`,一条龙脚本 `scripts/train_distill.sh`
- 老师输出许可:DeepSeek 开放平台服务条款 4.2 明确允许用输出训练其他模型(含蒸馏),输出权利归用户

## 部署(推荐配置,与评测协议一致)

```bash
vllm serve fang04/medforge-qwen3.5-4b-distill --max-model-len 36864
```

请求参数用 Qwen3.5 官方卡思考模式的采样参数,**不要用贪心**:

```python
client.chat.completions.create(
    model="...", messages=msgs,
    temperature=1.0, top_p=0.95, presence_penalty=1.5, max_tokens=32768,
    extra_body={"top_k": 20, "min_p": 0.0},
)
```

本模型在贪心下也不复读,但采样臂三卷都略优;`</think>` 之后的最后一行是「答案:X」,可直接用正则抽取。

## 局限

- 单随机种子、单次训练;CMB-val 仅 280 题(最小可检出差异约 7pp);MedXpertQA 无增益
- 教材未做「答案对但推理不成立」的人工抽检(老师判对率 94%,规则层只核答案字母)
- 老师与判卷兜底模型同厂(DeepSeek);严格口径由规则层判定(声明率 ≥ 99%),兜底占比 <1%
- 不会说「不知道」:2000 题只主动弃权 5 题;弃权训练是下一步
- 中文考试题为主要目标;非医疗建议

完整代码、评测报告与全部负结果:https://github.com/yiongq/medforge
