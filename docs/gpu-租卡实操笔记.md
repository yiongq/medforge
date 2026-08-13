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
- `pkill -f 脚本名` 会杀掉自己的 ssh 会话(命令串里含同名),模式写成 `脚本名[.]sh`

## 计费

- 按量计费,关机即停;连续关机 15 天自动释放(数据清空)
- 一切可重建:代码在 git、模型/数据集脚本重下、密钥本地留档——释放无损失
- W1 实际学费:两台机约 15 元(其中 12 元是 550 驱动那台的无效试错)
