"""GRPO 奖励插件:选择性预测(selective prediction)奖励,注册名 ``medforge_selective``。

用法(configs/grpo_qwen35_4b_lora.yaml 已经这样写了):

    external_plugins: [src/medforge/train/grpo_reward.py]
    reward_funcs: [medforge_selective]

ms-swift 4.5.2 的加载路径(读 wheel 源码核对过,不是猜的):
  1. BaseArguments.__post_init__ → BaseArguments._import_external_plugins()
     (swift/arguments/base_args/base_args.py:142)
  2. → swift.utils.utils.import_external_file(file_path)(swift/utils/utils.py:401):
     把本文件所在目录 insert 进 sys.path 再 importlib.import_module("grpo_reward")
     —— 注意它是按**顶层模块名**导入的,不是 medforge.train.grpo_reward。
     所以本文件必须能在「medforge 没装进 swift venv」的前提下自己找到 medforge 包(见下面的 sys.path 兜底)。
  3. 导入的副作用就是注册:本文件末尾把类塞进 swift.rewards.orms 字典
     (registry 定义在 swift/rewards/orm.py:459)。
  4. GRPOTrainer._prepare_rewards → swift/rlhf_trainers/utils.py:2192 resolve_reward_funcs
     以 ``orms[name](args=args)`` 实例化。

奖励口径(这是本插件存在的全部理由):

    答对          +1.0
    主动弃权       0.0   (MEDFORGE_GRPO_ABSTAIN_REWARD)
    答错          -1.0   (MEDFORGE_GRPO_WRONG_REWARD)
    没写完 / 没声明 -1.0   (MEDFORGE_GRPO_UNFINISHED_REWARD)

为什么弃权给 0 而答错给 -1:这套数值让「弃权」恰好在 p(答对) < 0.5 时成为最优动作。
期望收益 = p·(+1) + (1-p)·(-1) = 2p - 1,与弃权的 0 相等当且仅当 p = 0.5;p < 0.5 时猜的期望为负,
弃权更划算。也就是说奖励表本身就在教模型「什么时候该说不确定」,而不是靠提示词求它别猜——
这正是 P2 裁决里「贪心 + 允许写不确定」那条臂失败的地方(prompt 给了弃权许可,模型并不会用)。
想换成更保守的策略(比如只在 p < 0.25 时才值得答)就把 wrong 调到 -3:阈值 = (1-wrong)/2 的倒推。
三个常量都能用环境变量覆盖,这样调阈值不用改代码、也不用重新提交配置。

未收尾同样罚 -1 而不是 0:第三阶段是接在弃权 SFT 之后的,「转圈到撞上限」与「答错」在交付上一样糟,
给 0 会让模型发现「写不完」比「答错」便宜,直接退化成永不收尾(W2 实测过 DPO 的这个退化方向)。

判分与评测同源:split_answer + extract 就是 medforge.verify 里评测在用的那两个函数,
不另写一套「训练用的宽松抽取」——训练奖励和考卷分数必须是同一把尺子。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# ms-swift 用 import_external_file 顶层导入本文件,而 medforge 未必装进了训练 venv
# (scripts/setup_train_env.sh 只装 ms-swift)。本文件在 src/medforge/train/ 下,parents[2] 就是 src/。
if __package__ in (None, ""):  # pragma: no cover - 仅在 ms-swift 侧的顶层导入时走到
    _SRC = Path(__file__).resolve().parents[2]
    if str(_SRC) not in sys.path:
        sys.path.insert(0, str(_SRC))

from medforge.verify.extract import extract
from medforge.verify.verifier import split_answer

CORRECT_REWARD = 1.0
ABSTAIN_REWARD = 0.0
WRONG_REWARD = -1.0
UNFINISHED_REWARD = -1.0

ENV_PREFIX = "MEDFORGE_GRPO_"


def _env_float(name: str, default: float) -> float:
    """环境变量覆盖;写坏了(非数字)就退回默认值而不是炸掉——训练跑到一半因为一个 typo 崩掉最贵。"""
    raw = os.environ.get(ENV_PREFIX + name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def normalize_letters(raw: object) -> frozenset[str]:
    """把 gold / 抽取值统一成字母集合:多选比集合,"CA" 与 "A、C" 与 "AC" 等价。"""
    if not isinstance(raw, str):
        return frozenset()
    return frozenset(c for c in raw.upper() if c.isalpha())


def _clean_options(options: object) -> dict[str, str] | None:
    """选项列的兜底清洗。

    数据集经 Arrow 往返后,选项数不齐的题会被补成 {"A": ..., "F": None} 这种带 None 的 struct;
    extract 只用 options 的**键集合**来判连写多选是否合法,带 None 的键会放宽本不该放宽的字母。
    另外允许 options 是 JSON 字符串,方便手工造数据集时偷懒。
    """
    if isinstance(options, str):
        import json

        try:
            options = json.loads(options)
        except ValueError:
            return None
    if not isinstance(options, dict):
        return None
    cleaned = {str(k): str(v) for k, v in options.items() if v is not None}
    return cleaned or None


def score_one(
    completion: str,
    solution: object,
    options: object = None,
    finish_reason: str | None = None,
    *,
    correct: float = CORRECT_REWARD,
    abstain: float = ABSTAIN_REWARD,
    wrong: float = WRONG_REWARD,
    unfinished: float = UNFINISHED_REWARD,
) -> float:
    """单条 completion 的奖励。四个分支与验证器的四态判分一一对应。"""
    if not isinstance(completion, str) or not completion:
        return unfinished
    # thinking=True:第三阶段的基座是思考型模型,没有 </think> 就是没收尾,
    # 不能从未收尾的思考流里刮答案(medforge.verify.verifier 的截断守卫,与评测同一口径)
    answer, unfinished_reason = split_answer(completion, finish_reason=finish_reason, thinking=True)
    if unfinished_reason is not None:
        return unfinished
    ext = extract(answer, True, options=_clean_options(options))
    if ext is None:
        return unfinished  # 写完了但没做出可抽取的声明:与没写完同罚,别教它含糊其辞
    if ext.kind == "abstain":
        return abstain
    if ext.kind != "choice":
        return unfinished
    gold = normalize_letters(solution)
    if not gold:
        return unfinished  # 数据侧缺金标:这条题给不出信号
    return correct if normalize_letters(ext.value) == gold else wrong


try:  # pragma: no cover - swift 只装在 GPU 机的训练 venv 里
    from swift.rewards import ORM as _ORM
except ImportError:  # pragma: no cover - 本机(无 ms-swift)跑单测时的占位基类

    class _ORM:
        def __init__(self, args=None, **kwargs):
            self.args = args


class MedforgeSelectiveReward(_ORM):
    """选择性预测奖励。ms-swift 以 ``orms['medforge_selective'](args=args)`` 实例化。

    调用约定(swift/rl_core/grpo_algorithm.py compute_rewards_per_func):
    ``reward_func(completions, **reward_kwargs)``,其中 completions 是本批各样本的 assistant 文本,
    reward_kwargs 里每个数据集列都是一条与 completions 等长的 list(RowPreprocessor.rows_to_batched),
    另外还带 finish_reason / is_truncated / trainer_state 等运行期列。
    """

    def __init__(self, args=None, **kwargs):
        super().__init__(args, **kwargs)
        self.correct = _env_float("CORRECT_REWARD", CORRECT_REWARD)
        self.abstain = _env_float("ABSTAIN_REWARD", ABSTAIN_REWARD)
        self.wrong = _env_float("WRONG_REWARD", WRONG_REWARD)
        self.unfinished = _env_float("UNFINISHED_REWARD", UNFINISHED_REWARD)

    @staticmethod
    def _column(value: object, n: int) -> list:
        """把一个 reward kwarg 归一成长度 n 的 list:缺列时补 None,标量时广播。"""
        if isinstance(value, list):
            return value + [None] * (n - len(value)) if len(value) < n else value
        return [value] * n

    def __call__(self, completions, solution=None, options=None, finish_reason=None, **kwargs) -> list[float]:
        n = len(completions)
        sols = self._column(solution, n)
        opts = self._column(options, n)
        fins = self._column(finish_reason, n)
        return [
            score_one(
                completions[i],
                sols[i],
                opts[i],
                fins[i] if isinstance(fins[i], str) else None,
                correct=self.correct,
                abstain=self.abstain,
                wrong=self.wrong,
                unfinished=self.unfinished,
            )
            for i in range(n)
        ]


REWARD_NAME = "medforge_selective"

try:  # pragma: no cover - 注册副作用:import 本文件即完成注册
    from swift.rewards import orms

    orms[REWARD_NAME] = MedforgeSelectiveReward
except ImportError:  # pragma: no cover - 本机无 ms-swift:只当普通模块用,单测照跑
    pass
