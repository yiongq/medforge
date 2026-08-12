"""统一样本 schema。

所有数据源(可验证题/考试题/蒸馏 CoT)进入管线前都归一到 Sample,
下游(去污染、SFT 构造、DPO 采样、评测)只认这一个形状——
这样换数据源只改 normalize 的 adapter,管线本身不动。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Sample:
    id: str                      # "{source}-{序号}",全局唯一,贯穿训练/评测/报告
    source: str                  # 数据源名,如 "med-o1-verifiable"
    question: str
    gold: str                    # 标准答案:选择题为选项字母(多选按字典序连写,如 "ACD"),开放题为答案文本
    options: dict[str, str] | None = None   # 选择题选项 {"A": "...", ...};开放题为 None
    cot: str | None = None       # 数据源自带的推理过程(蒸馏 SFT 数据才有)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def is_choice(self) -> bool:
        return self.options is not None

    def render_question(self) -> str:
        """题面渲染:选择题把选项拼进题干,保证模型看到的和人看到的一致。"""
        if not self.options:
            return self.question
        opts = "\n".join(f"{k}. {v}" for k, v in sorted(self.options.items()))
        return f"{self.question}\n{opts}"
