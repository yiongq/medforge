# MedForge

把开源小模型锻造成医疗推理特长生:**可验证数据 → SFT → 验证器驱动 DPO/GRPO → 对照评测 → 生产级部署**,全流程开源、全数字可复现。

> ⚠️ 本项目是研究与工程实践,输出不构成任何医疗建议。依据国内规定,AI 不得用于自动诊断与处方。

## 结果(施工中,W2 出数)

| 配置 | CMB-val¹ | CMExam | MedXpertQA-Text | 幻觉(MedHallu 子集) |
|---|---|---|---|---|
| Qwen3.5-4B(base,v2 抽样卷) | 59.6% | 70.8% | 25.1% | – |
| + SFT(2024 蒸馏教材) | 47.5% ↓ | 59.3% ↓ | 13.6% ↓ | – |
| + SFT(2025 R1 蒸馏教材) | 53.9% ↓ | 67.5% ↓ | 13.6% ↓ | – |
| + DPO(基座自采,验证器驱动) | 56.1% | 70.7% | 24.4% | – |
| + GRPO(计划) | – | – | – | – |
| Qwen3.5-9B 定稿复跑 | – | – | – | – |

*上表为宽口径,已知含撞上限后从复读段刮出的分,见下方 2026-09-03 订正。¹ CMB-val 是 CMB 官方的 dev/few-shot 池(n=280),不是榜单卷 CMB-test(11,200 题),数字与 CMB leaderboard 不可比。*

**W2 核心发现**([完整报告](reports/w2-post-training-ablation.md)):对思考型基座(Qwen3.5),蒸馏 SFT 全线降分——换更强的老师(2024 GPT-4o → 2025 R1)能大幅止损(主力卷 -11.5pp → -3.3pp),但困难卷两代教材同伤(25.1% → 13.6%),表明「抄外部笔记」本身即破坏原生深推理,与教材质量无关。协议注记:v2 = 8192 tokens + 固定种子抽样卷(cmexam 2000 / medxpertqa 1000 / cmb 全量),v1 全量存档见 reports/runs/base/。DPO 三卷均与基座持平(差值落在置信区间内),但逐题转移矩阵显示 CMExam 有 413/2000 题答案翻转(修好 206、弄坏 207)——训练确实改变了模型,方向却近乎随机。

**2026-09-03 订正**([严格口径重算](reports/usability.md)):上表是宽口径,把「撞上 token 上限、陷入复读、从未写出结论」的答卷也按末段刮出的字母计了分,而四个配置的未收尾率不在一个量级(CMExam:基座 25.7%、旧教材 SFT 0.3%、R1 教材 SFT 17.5%、DPO 33.1%;MedXpertQA:65.4 / 1.7 / 44.9 / 74.5%)。按「写完 ∧ 结论可抽出 ∧ 答对」重算:CMExam 上 R1 教材 SFT **62.2% vs 基座 59.2%**(配对差值 +3.0pp,95% CI [+0.7, +5.2],McNemar p=0.011;9 项比较做 BH 校正后成立、Holm 控 FWER 下不成立),这是唯一在严格口径下为正的比较,衡量的是交付可靠性而非答题能力——不设守卫时基座反而领先 3.7pp(p=0.002)。困难卷「25.1% → 13.6%」在严格口径下消失(两个 SFT 臂都不低于基座),上段的机制推论失去支柱;DPO 三卷严格口径均显著低于基座——它学到的是更少写完,不是答得更对。判分链已加截断守卫并落盘 finish_reason(协议 v3),上表待 v3 重跑后改定。

**2026-09-04 解码裁决**([完整报告](reports/p2-decoding-arms.md)):同一份基座权重、不训练、只换解码方式,五条对照臂:

| 基座 Qwen3.5-4B,严格口径 | CMExam | CMB-val | MedXpertQA | 收尾率 |
|---|---|---|---|---|
| 贪心 8192(W2 协议) | 59.6% | 46.4% | 10.2% | 33~74% |
| **官方采样参数 + 32768** | **74.0%** | **60.0%** | **25.1%** | 100% |
| 贪心 + 撞上限时强写「</think> 答案:」续写 32 token | 73.5% | 57.9% | 27.4% | 100% |
| 贪心 + 允许写「不确定」 | 57.6% | 45.7% | 9.9% | 29~70% |
| 贪心 + 预算放大到 32768 | 63.8% | 49.3% | 13.9% | 46~82% |

W2 成绩表测的是解码协议不是模型:官方明令禁止的贪心解码让思考型基座转圈交不了卷,预算翻四倍也救不回(18% 的题写满 32768 仍在循环),而一条 32 token 的续写补丁就能把分数拉回 74%。
训练臂尚未在 v3 协议下重评,上面的 W2 表只作 v2 协议下的历史数字保留;新的比较基线是「基座 + 强制收尾」,任何微调要先赢过它。

**2026-09-04 训练线在 v3 下重评 + 蒸馏 2.0**([完整报告](reports/distill-2.0.md) · [对照表](reports/usability-v3-train.md)):

| 严格口径,v3 采样,同批题配对 | CMExam(域内) | CMB-val(迁移) | MedXpertQA(迁移) | 每题 token 均值 |
|---|---|---|---|---|
| 基座 | 74.0% | 60.0% | 25.1% | 5.2k / 5.7k / 8.4k |
| DPO(自采样,8 月权重) | 74.7% | 58.9% | 26.1% | ≈ 基座 |
| **蒸馏 2.0**(DeepSeek 老师 · CMExam 训练集 · 五道闸门 · 修好的配置) | **80.4%** (+6.5, p<10⁻⁴) | **65.7%** (+5.7, p=0.04) | 26.4% (持平) | **1.8k / 2.3k / 4.7k** |
| DeepSeek V4-flash(老师,参照) | 93.2% | 84.3% | 47.0% | 0.6k / 1.3k / 5.4k |

DPO 在正确尺子下是真持平;蒸馏 2.0 是项目训练线第一个正结果——同一个 4B、不到三小时训练,主力卷 +6.5pp、迁移卷同向、成本降到 1/3。8 月失败的不是「蒸馏」这条路,是老师、题源、格式、长度、尺子五处同时错。

**部署侧**(RTX 5090 · vLLM · [完整报告](reports/deployment.md)):FP8 相比 BF16 峰值吞吐
5,810 vs 4,822 tok/s(+20%),且并发 64 时首字延迟更低(369 vs 455 ms)——同一份权重,仅数值精度不同。
七档并发(1→64)全程零失败,吞吐随并发近线性增长,说明单卡远未饱和。

## 产物

| 在哪 | 是什么 |
|---|---|
| [🤗 fang04/medforge-qwen3.5-4b-distill](https://huggingface.co/fang04/medforge-qwen3.5-4b-distill)([模型卡](docs/model-card-distill.md)) | **二代模型**:Qwen3.5-4B + 蒸馏 2.0,CMExam 严格口径 80.4%、每题 token 1/3 |
| [🤗 fang04/medforge-qwen3.5-4b-dpo](https://huggingface.co/fang04/medforge-qwen3.5-4b-dpo) | 一代模型:Qwen3.5-4B + 验证器驱动 DPO(v3 重评三卷与基座持平) |
| [🤗 fang04/medforge-artifacts](https://huggingface.co/datasets/fang04/medforge-artifacts) | 自采样原始解法(10 GPU 小时)、偏好对、判卷标签缓存、四方案完整答卷 |
| **[medforge.yiongspace.com](https://medforge.yiongspace.com)** | 在线实验台:同题四方案并排、成绩板、压测曲线、live 现场提问 |
| `web/` | 上述站点源码(Vite + React,零运行时依赖) |
| `reports/` | 去污染报告 · 验证器校准报告 · 各 run 判分与汇总 |

## 方法

复刻 [HuatuoGPT-o1](https://github.com/FreedomIntelligence/HuatuoGPT-o1)(ACL 2025)的核心思路并缩小到单卡规模:

1. **可验证医学题**(4 万道,开放题+标准答案)做 SFT 底料
2. **验证器**(规则抽取 + LLM 兜底;200 题校准集混合一致率 96.5%,2026-09 按选择题工作分布补校准 LLM 层 98.6%、κ 0.97)自动判卷
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

### 判卷后端(验证器 LLM 兜底 / 代理标注 / 参考臂)

`cp .env.example .env` 后填,两条后端二选一(`src/medforge/env.py` 读根目录 `.env`):

| 变量 | 说明 |
| --- | --- |
| `MEDFORGE_JUDGE_PROVIDER` | `openai`(默认)/ `claude-code` |
| `MEDFORGE_JUDGE_BASE_URL` / `_API_KEY` | 仅 `openai` 需要;任何 OpenAI 兼容端点 |
| `MEDFORGE_JUDGE_MODEL` | 两者都要;claude-code 下写全名(如 `claude-sonnet-5`),且必须与代理标注模型不同 |
| `MEDFORGE_JUDGE_EFFORT` | 可选,claude-code 的扩展思考档位 `low\|medium\|high\|xhigh\|max` |

`claude-code` 走本机已登录的 Claude Code CLI(`claude auth login`,用订阅额度,不需要 API key;
子进程环境会清掉 `ANTHROPIC_API_KEY` 等变量,免得悄悄改走按量计费),
评测臂另有 `--provider claude-code --effort high`;代价、口径妥协与验证方式见
[docs/claude-code-provider.md](docs/claude-code-provider.md)(先跑
`uv run python -m medforge.verify.claude_code --model claude-sonnet-5` 冒烟)。

> 蒸馏教师与 DPO 仲裁不走这条路:Anthropic 消费者条款禁止用 Claude 输出训练其他模型,
> `data/build_distill.py` 的教师、`data/build_dpo.py` 的偏好对仲裁继续用 DeepSeek(其条款 §4.2 允许蒸馏)。
> 判卷/标注/评测是测量,可以用 Claude;凡是直接产出训练教材的一步都不行。

## 路线图

- [x] W1a 骨架:数据下载与归一 / 验证器(规则层+LLM 兜底+校准 CLI)/ 去污染字面层 / 报告层 + 单测,经双路对抗审查修复
- [x] W1b 验证器校准(96.5%)+ 去污染真报告 + base 三卷底分落盘(两台租卡实战,踩坑笔记见 docs/)
- [x] W2 前置:协议 v2 重跑基线(✅)· 去污染 embedding 层(❌ 未做,只有字面层)· MedHallu 评测协议(❌ 未做)
- [x] W2 SFT → DPO 两档对照,4B 全流程(✅ 负结果已发表)· GRPO(❌ 未做,见 P2 裁决:先修测量制度)
- [x] W3 部署(vLLM BF16/FP8 压测 ✅;监控面板 ❌)+ 前台(同题四方案回放 + live ✅)· 9B 定稿复跑(❌ 未做)
- [x] W2 审查 + 协议 v3:截断守卫 / 四态判分 / 配对检验 / 解码裁决(✅ 2026-09)
- [x] 训练臂在 v3 协议下重评(DPO ✅ 持平;旧 SFT 无权重,以蒸馏 2.0 替代 ✅ +6.5pp)
- [ ] 蒸馏 2.0 上线(live 模式 / 部署配置)· 弃权训练(数据与配置已就位,待训) · 多 seed
- [ ] 二期 医疗工具调用模块 · 医疗 VQA LoRA 分支

### 训练线的下一段:弃权(第二阶段 SFT)

蒸馏 2.0 之后模型的问题不再是「答不完」,而是**不会也照答**——医疗场景里一个自信的错答比一句
「不确定」贵得多。做法是 R-Tuning 式的拒答微调:让蒸馏模型在一批没训过的题上自采样 K 次,
用严格可用协议(收尾 ∧ 声明 ∧ 判对)把题分成「K 次全对」和「一次没对」两堆,前者原样保留、
后者把它自己的思考接上一句过渡改写成「答案:不确定」,半对半错的题一律丢掉——弃权必须落在
这个模型真的不会的题上,否则教的是随机拒答。数据构造见
[`src/medforge/data/build_abstain.py`](src/medforge/data/build_abstain.py),训练配置
[`configs/sft_abstain_qwen35_4b_lora.yaml`](configs/sft_abstain_qwen35_4b_lora.yaml)(基座 = 蒸馏合并权重),
验收不看准确率而看选择性预测口径的覆盖率 / 选择性准确率 / 弃权精度召回,由
[`medforge.eval.abstain_report`](src/medforge/eval/abstain_report.py) 与训练前的同模型 run 配对出表。
