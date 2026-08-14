---
license: apache-2.0
base_model: Qwen/Qwen3.5-4B
language:
  - zh
  - en
tags:
  - medical
  - dpo
  - qwen3.5
  - post-training
library_name: transformers
pipeline_tag: text-generation
---

# MedForge · Qwen3.5-4B-DPO(验证器驱动的自采样偏好学习)

在 Qwen3.5-4B 上做医疗推理后训练的消融实验产物。**训练信号全部来自模型自己的采样结果**:
同题多次采样 → 经校准的验证器按标准答案判对错 → 对/错解配成偏好对 → DPO。
整个过程不引入任何外部模型写的推理示范。

> ⚠️ 研究与工程实践产物,**不构成任何医疗建议**;依据中国相关规定,AI 不得用于自动诊断与处方。

## 为什么是这条路线(实验结论)

同一基座、同一套考卷、同一评测协议下,我们对比了三种后训练配方:

| 配方 | CMExam(n=2000) | CMB-val(n=280) | MedXpertQA(n=1000) |
|---|---|---|---|
| Qwen3.5-4B(原装基座) | **70.8%** | **59.6%** | **25.1%** |
| + SFT · 2024 年 GPT-4o 蒸馏 CoT | 59.3%(−11.5) | 47.5%(−12.1) | 13.6%(−11.5) |
| + SFT · 2025 年 R1 蒸馏 CoT | 67.5%(−3.3) | 53.9%(−5.7) | 13.6%(−11.5) |
| **+ DPO · 自采样(本模型)** | 见下 | 见下 | 见下 |

**核心发现:对 thinking-native 基座,蒸馏 SFT 会降智。** 换更强的老师(GPT-4o → R1)能大幅止损
(主力卷 −11.5pp → −3.3pp),但困难卷两代教材同摔至 13.6% ——说明伤害不只来自教材质量,
而来自「模仿外部推理风格」这个动作本身:它覆盖了基座原生的长链思考。
自采样偏好学习是唯一不引入外部风格的路线,因而是本项目的定稿方案。

## 训练配置

- 基座:`Qwen/Qwen3.5-4B`,LoRA 全层挂载(r=64, α=128, lr=5e-6, 1 epoch)
- 偏好数据:约 1,000 对,源自 3,000 道可验证医学题(`FreedomIntelligence/medical-o1-verifiable-problem`)
  的自采样(temperature 1.0, k=6, max_tokens 8192)
- 判分:规则层抽取答案声明 + LLM 仲裁兜底;验证器上岗前经 200 题人工校准,一致率 96.5%
- 去污染:训练题池对全部评测集做字符 10-gram 查重,污染样本剔除,方法与数字公开
- 框架:ms-swift · 单卡 A800-80G

## 用法

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "fang04/medforge-qwen3.5-4b-dpo"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map="auto")

msgs = [{"role": "user", "content": "患者男性 65 岁,突发胸骨后压榨性疼痛 3 小时,心电图 V1-V4 导联 ST 段抬高。最可能的诊断是什么?"}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
out = model.generate(ids, max_new_tokens=8192)
print(tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True))
```

vLLM 部署:`vllm serve fang04/medforge-qwen3.5-4b-dpo`

## 局限

- 评测为固定种子抽样卷(CMExam 2000 / MedXpertQA 1000 / CMB-val 全量),非全量榜单口径
- 单随机种子、单次训练,未做多种子重复
- 判分的「人工校准」实为强模型代理标注 + 人工抽检,非纯人工
- 中英混合训练数据,中文考卷为主要目标

完整代码、评测报告与全部负结果:https://github.com/yiongq/medforge
