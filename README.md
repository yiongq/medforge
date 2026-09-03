# MedForge

把开源小模型锻造成医疗推理特长生:**可验证数据 → SFT → 验证器驱动 DPO/GRPO → 对照评测 → 生产级部署**,全流程开源、全数字可复现。

> ⚠️ 本项目是研究与工程实践,输出不构成任何医疗建议。依据国内规定,AI 不得用于自动诊断与处方。

## 结果(施工中,W2 出数)

| 配置 | CMB | CMExam | MedXpertQA-Text | 幻觉(MedHallu 子集) |
|---|---|---|---|---|
| Qwen3.5-4B(base,v2 抽样卷) | 59.6% | 70.8% | 25.1% | – |
| + SFT(2024 蒸馏教材) | 47.5% ↓ | 59.3% ↓ | 13.6% ↓ | – |
| + SFT(2025 R1 蒸馏教材) | 53.9% ↓ | 67.5% ↓ | 13.6% ↓ | – |
| + DPO(基座自采,验证器驱动) | 56.1% | 70.7% | 24.4% | – |
| + GRPO(计划) | – | – | – | – |
| Qwen3.5-9B 定稿复跑 | – | – | – | – |

*上表为宽口径,已知含撞上限后从复读段刮出的分,见下方 2026-09-03 订正。*

**W2 核心发现**([完整报告](reports/w2-post-training-ablation.md)):对思考型基座(Qwen3.5),蒸馏 SFT 全线降分——换更强的老师(2024 GPT-4o → 2025 R1)能大幅止损(主力卷 -11.5pp → -3.3pp),但困难卷两代教材同伤(25.1% → 13.6%),表明「抄外部笔记」本身即破坏原生深推理,与教材质量无关。协议注记:v2 = 8192 tokens + 固定种子抽样卷(cmexam 2000 / medxpertqa 1000 / cmb 全量),v1 全量存档见 reports/runs/base/。DPO 三卷均与基座持平(差值落在置信区间内),但逐题转移矩阵显示 CMExam 有 413/2000 题答案翻转(修好 206、弄坏 207)——训练确实改变了模型,方向却近乎随机。

**2026-09-03 订正**([严格口径重算](reports/usability.md)):上表是宽口径,把「撞上 token 上限、陷入复读、从未写出结论」的答卷也按末段刮出的字母计了分,而四个配置的未收尾率不在一个量级(CMExam:基座 25.7%、旧教材 SFT 0.3%、R1 教材 SFT 17.5%、DPO 33.1%;MedXpertQA:65.4 / 1.7 / 44.9 / 74.5%)。按「写完 ∧ 结论可抽出 ∧ 答对」重算:CMExam 上 R1 教材 SFT **62.2% vs 基座 59.2%**(配对检验 p=0.011),这是唯一在严格口径下显著为正的比较,衡量的是交付可靠性而非答题能力——不设守卫时基座反而领先 3.7pp(p=0.002)。困难卷「25.1% → 13.6%」在严格口径下消失(两个 SFT 臂都不低于基座),上段的机制推论失去支柱;DPO 三卷严格口径均显著低于基座——它学到的是更少写完,不是答得更对。判分链已加截断守卫并落盘 finish_reason(协议 v3),上表待 v3 重跑后改定。

**部署侧**(RTX 5090 · vLLM · [完整报告](reports/deployment.md)):FP8 相比 BF16 峰值吞吐
5,810 vs 4,822 tok/s(+20%),且并发 64 时首字延迟更低(369 vs 455 ms)——同一份权重,仅数值精度不同。
七档并发(1→64)全程零失败,吞吐随并发近线性增长,说明单卡远未饱和。

## 产物

| 在哪 | 是什么 |
|---|---|
| [🤗 fang04/medforge-qwen3.5-4b-dpo](https://huggingface.co/fang04/medforge-qwen3.5-4b-dpo) | 定稿模型权重(Qwen3.5-4B + 验证器驱动 DPO) |
| [🤗 fang04/medforge-artifacts](https://huggingface.co/datasets/fang04/medforge-artifacts) | 自采样原始解法(10 GPU 小时)、偏好对、判卷标签缓存、四方案完整答卷 |
| **[medforge.yiongspace.com](https://medforge.yiongspace.com)** | 在线实验台:同题四方案并排、成绩板、压测曲线、live 现场提问 |
| `web/` | 上述站点源码(Vite + React,零运行时依赖) |
| `reports/` | 去污染报告 · 验证器校准报告 · 各 run 判分与汇总 |

## 方法

复刻 [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)(ACL 2025)的核心思路并缩小到单卡规模:

1. **可验证医学题**(4 万道,开放题+标准答案)做 SFT 底料
2. **验证器**(规则抽取 + LLM 兜底,先经 200 题人工校准)自动判卷
3. 同题采样多解 → 验证器判对错 → 配成偏好对 → **DPO**;**GRPO** 作对照组
4. **EvalScope** 跑 CMB / CMExam / MedXpertQA,自写分析层出 Wilson 95% CI 对照表
5. 训练数据对全部评测集做去污染并公开方法与数字:字符 10-gram 字面层已实现,embedding 语义层 W1b 接入
6. **vLLM** 部署(FP8 主线 + AWQ 对照 + multi-LoRA 热加载)+ Prometheus/Grafana + 压测报告

技术选型的完整依据与放弃项:[docs/adr/001](docs/adr/001-技术选型与依据.md)。

## 仓库结构

```
configs/            实验配置(改配置=做实验)
src/medforge/
  data/             下载 → 归一 → 去污染 → 出训练数据
  verify/           验证器:答案抽取 + LLM 兜底判分
  eval/             EvalScope 之上的对照分析与报告
  tools/            (二期)医疗工具调用
  serve/            (W3)部署、压测、监控
scripts/            租卡环境 bootstrap、训练、评测入口
reports/            评测/压测报告(进 git,数字可追溯)
docs/adr/           架构决策记录
docs/教材/          零基础讲义(完工后配套)
```

## 快速开始

```bash
git clone <repo> && cd medforge
uv sync
uv run pytest                                   # 数据管线与验证器单测
uv run python -m medforge.data.download         # 拉取数据集(国内可加 HF_ENDPOINT=https://hf-mirror.com)
```

训练/评测在租用 GPU 上进行:`scripts/autodl_bootstrap.sh` 一键装环境,配置见 `configs/`。

## 路线图

- [x] W1a 骨架:数据下载与归一 / 验证器(规则层+LLM 兜底+校准 CLI)/ 去污染字面层 / 报告层 + 单测,经双路对抗审查修复
- [x] W1b 验证器校准(96.5%)+ 去污染真报告 + base 三卷底分落盘(两台租卡实战,踩坑笔记见 docs/)
- [ ] W2 前置:协议 v2 重跑基线 · 去污染 embedding 层 · MedHallu 评测协议
- [ ] W2 SFT → DPO → GRPO 三档对照,4B 全流程
- [ ] W3 部署(vLLM/FP8/压测/监控)+ 前台(同题三模型对比,回放+live 双模式)+ 9B 定稿复跑
- [ ] 二期 医疗工具调用模块 · 医疗 VQA LoRA 分支
