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
