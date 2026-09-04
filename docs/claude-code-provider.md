# Claude Code provider:用订阅额度当推理后端

项目里三个「要调外部大模型」的角色可以切到本机已登录的 Claude Code CLI,走 Max 订阅额度,不用 API key:

| 角色 | 代码位置 | 怎么切 |
| --- | --- | --- |
| LLM 兜底判卷 | `verify/verifier.py::verify_by_llm` | `MEDFORGE_JUDGE_PROVIDER=claude-code` |
| 代理标注 proxy_correct | `verify/sample_calibration.py::label_proxy` | 同上,或 `--provider claude-code` |
| 参考/API 评测臂 | `eval/run.py` | `--provider claude-code` |

底层是 `verify/claude_code.py`:把一次 headless 调用(`claude -p ... --output-format json`)包成同步函数。

**两个角色不在此列,而且不许切:蒸馏教师、DPO 偏好对仲裁。** Anthropic 消费者条款禁止用 Claude 的
输出训练其他模型;`data/build_distill.py` 的教师(题解直接进 SFT 教材)与 `data/build_dpo.py` 的
LLM 仲裁(标签直接决定哪条解是 chosen、哪条是 rejected,即偏好数据集本身)都继续用 DeepSeek
(其条款 §4.2 明确允许蒸馏)。`build_dpo` 在开跑前显式拒绝 `MEDFORGE_JUDGE_PROVIDER=claude-code`。
分界线是「产出训练教材」还是「量一张已经考完的卷」:判卷、标注、评测属于后者,可以用。

## 一、登录

```bash
claude auth login     # 用 Max 账号登录,登录态存在本机(注意是 `auth login`,没有顶层 `claude login`)
claude auth status    # 最便宜的体检:先看登录态,再花额度
uv run python -m medforge.verify.claude_code --model claude-sonnet-5   # 冒烟:打一次最小请求
```

冒烟会打印解析后的文本、结构化输出与 usage。报「退出码 1 / not logged in」就是登录态没起来。

模型 ID 要写全:`claude-opus-5` / `claude-sonnet-5` / `claude-haiku-4-5-20251001`。
裸别名(`opus` / `sonnet` / `haiku`)被 CLI 钉在旧一代,拿到的不是你以为的模型。

## 二、环境变量

| 变量 | 何时需要 | 说明 |
| --- | --- | --- |
| `MEDFORGE_JUDGE_PROVIDER` | 总是 | `openai`(默认,保持历史行为)/ `claude-code` |
| `MEDFORGE_JUDGE_MODEL` | 总是 | claude-code 下写全名,如 `claude-sonnet-5`;必须 ≠ 代理标注模型 |
| `MEDFORGE_JUDGE_BASE_URL` / `_API_KEY` | 仅 `openai` | claude-code 下完全不需要 |
| `MEDFORGE_JUDGE_EFFORT` | 可选 | `low`/`medium`/`high`/`xhigh`/`max`,扩展思考预算档位 |

模板见 `.env.example`。写了非法的 provider 值不会报错,退回 `openai`——判分链的契约是「永不抛」。
但**非法的 effort、以及 PATH 上没有 `claude`**,都会让每一条判卷失败退化成弃权、而弃权按错算,
所以这两件由 `missing_judge_env()` 在 eval / DPO 入口开跑前拦下(直接 exit 2),不留到跑完才发现。

## 三、实测代价:额度才是瓶颈,不是延迟

- 每次调用有约 **31.6k 输入 token** 的基线上下文(CLI 自带的系统提示)。首次调用记在
  `cache_creation_input_tokens`,随后连续调用命中缓存(`cache_read ≈ 31.6k`、`cache_creation = 0`),
  单次约 **5 秒**。(2026-09-05 用本模块的参数实测 claude-sonnet-5,不是别的 flag 组合下的数字。)
- 所以 per-call 开销可接受,但必须限流:并发 **4~8** 就够,再高只是排队,还更容易撞
  **Max 的 5 小时额度窗口**。窗口用完就得等,评测跑到一半断在这里比慢更难受——
  长卷建议分批跑,断点续跑本来就只补空缺。
- 调用**不落会话文件**、**不带工具**、**不读项目配置**(`--tools "" --setting-sources "" --disable-slash-commands
  --no-session-persistence`),cwd 指向临时空目录——否则仓库的 `CLAUDE.md` 会被当上下文塞进判卷提示词,
  判分口径就被环境悄悄改写了。
- 进程环境里必须清掉 `CLAUDECODE` 与所有 `CLAUDE_*`,否则 CLI 判定自己被嵌套调用直接拒绝;
  `ANTHROPIC_API_KEY` / `_AUTH_TOKEN` / `_BASE_URL` 也一并清掉——留着的话 CLI 会改走按量计费,
  而这条 provider 的全部意义是用订阅额度,文档不能和实际计费口径打架。
- 评测臂对 `--concurrency` 有上限:`> 8` 会被降到 8 并打印提示(每条请求是一个 CLI 进程)。
  CLI 这条路没有 SDK 的 `max_retries`,所以生成臂内建 2 次重试、判卷内建 1 次,免得一次抖动
  就把整臂的失败率顶过 `MAX_FAIL_RATE` 而不出表。

## 四、评测臂的两处口径妥协(必须知道)

用 `--provider claude-code` 跑评测臂时,有两件事和 OpenAI 兼容端点不一样,`run_meta.json` 里都如实记着:

1. **采样参数记 null。** CLI 不暴露 temperature/top_p/top_k/min_p/presence_penalty/seed/max_tokens,
   所以这些键在指纹里是 `null`,而不是记一个没发出去的数字。指纹不许说谎。
2. **`thinking` 强制记 `off`。** CLI 的 JSON 结果只有最终答案,拿不到扩展思考的文本,存档答卷里永远没有
   `</think>`;若按 `on` 记账,截断守卫会把每一题都判成「未收尾」。思考其实是开着的(由 `--effort` 控预算,
   默认 `high`),`completion_tokens` 取 `usage.output_tokens`(含思考 token)。
   `finish_reason` 恒为 `"stop"`——CLI 不区分「写满预算」,别把它当端点报告的 stop 来读。

`provider` 与 `effort` 都进了 `PROTOCOL_KEYS`:同一个模型名经 CLI 与经 API 拿到的作答不是一回事,
混进同一个 run 目录会被 `check_protocol` 拒绝。这两个键在本分支之前的 `run_meta.json` 里不存在,
所以 `check_protocol` 给它们定了 legacy 缺省(`provider="openai"`、`effort=null`,见
`PROTOCOL_LEGACY_DEFAULTS`):存量目录照常续跑并在续跑时补写这两个键,拿 claude-code 去续老目录仍然被拦。

**budget-forcing 模式与这个 provider 不兼容**(要往 `/v1/completions` 灌裸 prompt 续写,CLI 没有这个接口),
`--extra-body`、`--endpoint` 同理:三者都在开跑前就退出,不会烧完额度才发现。

## 五、常用命令

```bash
# 判卷员切到 Claude(.env 里配好 PROVIDER/MODEL 即可,评测的 LLM 兜底判分就跟着走;
# DPO 构造不跟着走——它会拒绝 claude-code,见开头的政策线)
uv run python -m medforge.eval.run --endpoint http://127.0.0.1:8000/v1 --model Qwen3.5-4B --run-name base

# 代理标注(第二意见,effort 默认 high;并发克制些)
# 模型必须与 MEDFORGE_JUDGE_MODEL 不同,否则量的是「它和自己的一致率」,直接 exit(2):
# 推荐判卷 claude-sonnet-5 + 代理标注 claude-opus-5
uv run python -m medforge.verify.sample_calibration --label-proxy \
  --provider claude-code --model claude-opus-5 --concurrency 4

# 参考臂:考同一套卷、同一份提示词
uv run python -m medforge.eval.run --provider claude-code --model claude-opus-5 \
  --run-name api-opus5 --sets cmexam --samples cmexam=300 --concurrency 4
```
