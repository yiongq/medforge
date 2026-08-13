# MedForge

把开源小模型锻造成医疗推理特长生:**可验证数据 → SFT → 验证器驱动 DPO/GRPO → 对照评测 → 生产级部署**,全流程开源、全数字可复现。

> ⚠️ 本项目是研究与工程实践,输出不构成任何医疗建议。依据国内规定,AI 不得用于自动诊断与处方。

## 结果(施工中,W2 出数)

| 配置 | CMB | CMExam | MedXpertQA-Text | 幻觉(MedHallu 子集) |
|---|---|---|---|---|
| Qwen3.5-4B(base,协议 v1¹) | 59.6% | 71.1% | 21.6% | – |
| + SFT | – | – | – | – |
| + SFT + DPO | – | – | – | – |
| + SFT + GRPO | – | – | – | – |
| Qwen3.5-9B 定稿复跑 | – | – | – | – |

¹ v1 = temp 0 / max_tokens 2048,思考型模型在难题上被截断,困难卷为保守下界;W2 起用协议 v2 重跑基线后同协议对比。详见 [reports/runs/base/summary.md](reports/runs/base/summary.md)。

部署侧:vLLM + FP8/AWQ 压测曲线(TTFT / 吞吐 vs 并发)见 `reports/`(W3)。

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
