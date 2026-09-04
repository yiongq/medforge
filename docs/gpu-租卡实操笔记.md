# GPU 租卡实操笔记(AutoDL,2026-08 实测)

W1 底分评测在两台 4090 上共踩 10+ 坑的完整复盘。每条都真实发生过;
下次租卡照此清单,10 分钟可达评测就绪状态。

## 选机器(最重要的一条)

**盯「驱动/CUDA」列,选 CUDA ≥ 13.0 的主机。** 2026 年的 PyTorch 默认按 CUDA 13 编译,
驱动 550(12.4)/570(12.8)的主机跑不了 PyPI 默认轮子:

- 驱动 550(CUDA 12.4):死路。cu12x 新轮子已停发,退旧版又不认 qwen3_5,别租
- 驱动 570(CUDA 12.8):可绕,但要 cu129 变体(注意**没有 cu128**,torch 2.13 只发 cu129/cu130):
  ```bash
  uv pip install vllm==0.27.1 --extra-index-url https://wheels.vllm.ai/0.27.1/cu129 --torch-backend=cu129
  ```
  两个参数缺一不可:漏 extra-index 则 vllm 本体还是 cu130 编译,运行时 undefined symbol;
  `--torch-backend` 别用 auto(可能探测到没货的 cu128 索引)
- 驱动 580+(CUDA 13.0):默认安装直接工作,选这档

镜像选基础镜像里 CUDA ≤ 主机的最高档即可——我们自建 venv,镜像里的 torch 不用。

## 环境安装的四条军规(bootstrap 已固化,此处记为什么)

1. **卸掉学术加速代理再装包**:它对所有流量生效且大文件必截断(实测 uv 的 Python 包、
   vLLM 400MB 轮子两次翻车);我们的源(阿里 PyPI/ModelScope/hf-mirror)全是国内直连,
   只有 git clone GitHub 需要它
2. **用镜像 conda 自带的 Python**(/root/miniconda3/bin/python3.12):非交互 shell 下
   conda 不激活、pip 不在 PATH;uv 自装 Python 又会被代理截断
3. **推理栈与训练栈分 venv**:ms-swift 与 vllm 同锅会让解析器连锁降级
   (实测 ms-swift 4.4.2 + vllm 0.14 + transformers 4.57,旧 transformers 不认 qwen3_5)
4. **锁文件是版本唯一事实源**:`uv run` 每次隐式 sync 会把包按锁降回去——
   bootstrap 里 pip 层升的版本会被悄悄回滚;要升版本改 `uv lock --upgrade-package`,不要只改安装层

## 2026-09-03 租 5090 D 新踩的坑(P2 解码裁决那次)

- **uv 官方安装器会卡死**:bootstrap 第 2 步 `curl astral.sh/uv/install.sh` 从 GitHub releases 下载 tar,这台机器上
  一个字节都下不动(12 秒 0 → 0)。改从阿里源装:`/root/miniconda3/bin/python3.12 -m pip install -i https://mirrors.aliyun.com/pypi/simple uv`,
  再 `ln -sf /root/miniconda3/bin/uv ~/.local/bin/uv`,重跑 bootstrap(幂等,`command -v uv` 会跳过安装)
- **评测集不进 git,上机要先传**:`data/raw/{cmexam.test,cmb.val,medxpertqa.test}.jsonl` 三份共 11MB,
  `scp` 上去比在机器上重跑 `medforge.data.download` 快得多(后者还要拉训练集)
- **系统盘只有 30G**:仓库、`.venv`、`models/` 都放 `/root/autodl-tmp`(数据盘),并 `export UV_CACHE_DIR=/root/autodl-tmp/uv-cache`
  (uv 缓存 300MB+,默认在系统盘 `~/.cache`)
- **pkill 自杀的第三种写法**:`pkill -f "eval_p2_arms[.]sh"` 本身安全,但同一条 ssh 命令里若还有一句
  `bash scripts/eval_p2_arms.sh ...`(比如「先杀再重启」),命令行里就出现了字面量,自己照样被杀(exit 255)。
  先杀后启动必须拆成两次 ssh,或者只杀 `medforge[.]eval[.]run` 让脚本因 set -e 自行退出
- **32k 预算的贪心臂要给 `--timeout 3600`**:复读跑满 32768 token 一条请求十来分钟,默认 300 秒超时按生成失败计,超 2% 整臂退出
- **vLLM 0.28 + Qwen3.5-4B 不需要任何额外参数**:`reasoning_parser=''`,content 里带 `</think>`;`--max-model-len 36864`
  在 32G 上 0.92 显存占用起得来,concurrency 32 下三卷吞吐约每 15 分钟 400 题(CMExam)
- 实际账单:5090 D ¥2.78/时,五条臂 + 环境约 9 小时

## 远程执行

- 模型给 vLLM 前先解析成本地路径(`modelscope.snapshot_download` 幂等):
  vllm 默认走 HF Hub,不认 MODELSCOPE_CACHE
- 后台任务:`(setsid nohup bash x.sh > x.log 2>&1 < /dev/null &)` 双重脱离,
  少任何一环 ssh 会话都可能挂住;机器上没有 tmux
- `pkill -f 脚本名` 会杀掉自己的 ssh 会话(命令串里含同名),模式写成 `脚本名[.]sh`;
  `pgrep -f` 同坑——把结果喂给 kill 前必须用 [x] 字符类写模式,否则第一个被杀的就是你自己
  (历次「神秘 exit 255 断线」全是这个原因,实锤于 PID 同时出现在三组匹配里)

## 计费

- 按量计费,关机即停;连续关机 15 天自动释放(数据清空)
- 一切可重建:代码在 git、模型/数据集脚本重下、密钥本地留档——释放无损失
- W1 实际学费:两台机约 15 元(其中 12 元是 550 驱动那台的无效试错)

## W2 GPU 日检查单(审查提示的实测项)

- vLLM 起 SFT 后模型时,确认返回的 content 是否含 `<think>` 段:若挂了 reasoning parser,
  think 会被剥进 reasoning_content,DPO 采到的 chosen/rejected 就和 SFT 教的思考格式不一致——
  必要时把 reasoning_content 拼回,保持一致
- LoRA 合并用固定路径:swift export --adapters <ckpt> --merge_lora true --output_dir output/sft_qwen35_4b_lora/merged
- build_dpo 默认要求 judge 已配置(fail-fast 会拦),GPU 机上记得放 .env

## W3 部署压测清单

一次租卡跑完(建议 4090/5090,半天足够):

```bash
bash scripts/autodl_bootstrap.sh                      # 推理栈
bash scripts/serve_bench.sh fang04/medforge-qwen3.5-4b-dpo "RTX 5090"
# 产物:reports/bench-{bf16,fp8}.json · reports/deployment.md · web/public/bench.json
```

- **live 模式要让浏览器连得上**:AutoDL 走「自定义服务」端口(把 vLLM 起在 6006),
  或本机 `ssh -L 8000:127.0.0.1:8000 -p <port> root@<host>` 建隧道后填 `http://127.0.0.1:8000/v1`
- vLLM 默认放行跨域,浏览器可直连,不需要额外反代
- 压测前确认没有别的进程占显存(`nvidia-smi`),否则 FP8 那一档会因残留显存起不来

## GRPO 单卡备忘(2026-09,32 GB 一张卡跑 colocate rollout)

跑法:`bash scripts/train_grpo.sh`(配置 `configs/grpo_qwen35_4b_lora.yaml`)。三件事和 SFT 不一样。

**一张卡要同时装训练和推理,靠的是分时不是分空间。** GRPO 每一步先 rollout(vLLM 采样 8 个解)再更新权重,
`vllm_mode: colocate` 让两者同卡。显存这样分:`vllm_gpu_memory_utilization: 0.4`(32 GB × 0.4 ≈ 12.8 GB 给 vLLM,
4B bf16 权重约 8 GB,剩下做 KV),其余留给训练侧。真正让它跑得起来的是三个开关:
`sleep_level: 1` 在训练步开始前让 vLLM 睡下交还显存,`offload_model` / `offload_optimizer` 在 rollout 期间
把训练权重和优化器状态挪到 CPU。三个里少任何一个,单卡必 OOM。
另外 `beta: 0.0` 不加载参考模型,又省下一整份 4B 权重——代价是没有 KL 约束,得靠 `log_completions` 盯格式有没有崩。

**vllm 必须装进训练 venv,这是「推理栈与训练栈分 venv」那条军规唯一的破例。** colocate 的 vLLM 是在训练进程内
`import vllm` 起来的,而 `setup_train_env.sh` 刻意只装了 ms-swift。`train_grpo.sh` 的第 0.5 步做这件事:
先 `import vllm` 探测,缺了才装,版本从推理 venv 里读出来对齐(那边是跑通过的组合,别在这里另挑一个赌一次),
装前装后各打印一次 torch/transformers/trl/peft/swift/vllm 版本。**装完必须验证 transformers 还认得 qwen3_5**
(脚本里查 `CONFIG_MAPPING_NAMES`,不认就当场中止)——这正是本文档上面记的那次连锁降级:
版本号看着都正常,坏的是架构解析,错误会推迟到加载权重甚至更晚才炸。回滚办法是重跑 `setup_train_env.sh` 重建 venv。

**奖励插件是按顶层模块名导入的。** ms-swift 4.5.2 的 `external_plugins` 走
`swift/utils/utils.py:import_external_file`:把插件文件所在目录 insert 进 `sys.path`,然后
`importlib.import_module("grpo_reward")`——不是 `medforge.train.grpo_reward`。所以 `medforge` 没装进 swift venv 时
插件自己要把 `src/` 补进 `sys.path`(已经写在 `grpo_reward.py` 顶部)。注册靠 import 副作用:
文件末尾把类塞进 `swift.rewards.orms` 字典,配置里的 `reward_funcs: [medforge_selective]` 就是这个键。

**参数名闸门和 SFT 那道一样,但要换 dataclass。** GRPO 的键散在
`RLHFArguments / GRPOArguments / GRPOArgumentsMixin / RolloutTrainerArgumentsMixin / VllmArguments` 五个 dataclass 里,
脚本走 `RLHFArguments.__mro__` 一次收全。写错的键 ms-swift 是静默忽略而不是报错,训完才发现某个开关没生效最贵。
