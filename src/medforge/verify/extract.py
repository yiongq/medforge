"""从模型输出中抽取最终答案(验证器的规则层)。

设计约束:宁可抽不出来(返回 None 交给 LLM 兜底),不要抽错——
抽错的答案会污染 DPO 偏好对,这比浪费一次 LLM 调用贵得多。
所以每条规则只认高置信的格式,模糊情况一律弃权。

W1a 审查修复记录(每条都有实测复现,见 tests/):
- 「B超/CT/X线/D二聚体」等拉丁字母开头的医学名词曾被误抽首字母 → 字母必须是独立 token
- 「不应选 C」的否定声明曾被当成答案且 last-wins 压过真声明 → 否定前缀不算声明
- 「选择」作普通动词曾误触发(「选择 A 的考生忽略了…」)→ 从触发词移除
- 中英混排「C, as ...」曾把 a 并进多选 → 延续段只认独立大写字母
- CMB 实测 6 选项、MedXpertQA 实测 10 选项(gold 含 I/J 各约 10%)→ 字母集放宽到 A-J
- \\boxed{\\text{C}} 抽不出白烧 LLM 调用 → 先剥一层 \\text 包装

W2 审查修复记录:
- 多选延续段的分隔符曾含换行:「最终答案:B\\n\\nC 选项是干扰项」被抽成 BC → 只认行内空白
- 评测提示词自己要求的连写格式「答案:ABD」抽不出 → 新增连写分支,但只认
  「触发词 + 2~10 个升序大写字母 + 行尾」,且字母须在选项集合内(DIC/ECG/CEA 这类缩写因不升序被拒)
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

# 独立 token 的选项字母:后面紧跟汉字/字母/数字的不算(排除 B超、CT、X线)
_L = r"[A-J](?![A-Za-z0-9\u4e00-\u9fff])"
_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_WRAP_RE = re.compile(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}")  # \boxed{\text{C}} 的内层包装
_TRIGGER = r"(?:最终答案|正确答案|答案|(?<![不别勿莫误错])故选|(?<![不别勿莫误错])应选)"
# 触发词:「选择」是普通动词不收;故选/应选带否定前缀(不应选/误选/别选)时不算声明
# 延续段分隔只认行内空白:跨行的字母是下一段正文,不是多选延续
_ANSWER_LINE_RE = re.compile(
    rf"{_TRIGGER}\s*(?:是|为|:|:)?\s*[\*\s「【\[(]*({_L}(?:[ \t]*[、,,和 \t][ \t]*{_L})*)"
)
# 连写多选「答案:ABD」:必须顶到行尾(允许尾随标点/星号/括号),否则「答案:ACEI 类药物」会被抽成 ACEI
_ANSWER_LINE_MULTI_RE = re.compile(
    rf"{_TRIGGER}\s*(?:是|为|:|:)?\s*[\*\s「【\[(]*([A-J]{{2,10}})[\*\s」】\])。.]*$",
    re.MULTILINE,
)
_EN_ANSWER_RE = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:)\s*[\*\s]*\(?([A-Ja-j])\)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_LETTERS_RE = re.compile(r"[A-Ja-j]")
# boxed 内容整体是「字母+分隔」才算选择题答案;含其他文字(如 \boxed{肺栓塞})不算
_BOXED_CHOICE_RE = re.compile(r"[A-Ja-j\s、,,和]+")


@dataclass
class Extracted:
    kind: str    # "choice" | "text"
    value: str   # choice: 规范化选项字母(多选按字典序连写,如 "ACD");text: 答案原文


def _norm_letters(raw: str) -> str:
    return "".join(sorted({c.upper() for c in _LETTERS_RE.findall(raw)}))


def _valid_multi(letters: str, options: Iterable[str] | None) -> bool:
    """连写多选的合法性:不重复、升序(ABD 是答案,DIC/ECG 是缩写)、都在选项集合内。"""
    if len(set(letters)) != len(letters) or letters != "".join(sorted(letters)):
        return False
    return options is None or set(letters) <= {k.upper() for k in options}


def extract_choice(output: str, options: Iterable[str] | None = None) -> Extracted | None:
    """选择题:只认显式的答案声明,且以最后一次声明为准(模型可能中途改口)。

    options 给出题目的选项字母集合时,连写多选分支会据此拒绝不在集合内的字母。
    """
    tail = _WRAP_RE.sub(r"\1", output[-2000:])  # 答案声明总在结尾段;截尾防长推理干扰
    candidates: list[tuple[int, str]] = []
    for m in _BOXED_RE.finditer(tail):
        if _BOXED_CHOICE_RE.fullmatch(m.group(1)):
            letters = _norm_letters(m.group(1))
            if letters:
                candidates.append((m.start(), letters))
    for m in _ANSWER_LINE_RE.finditer(tail):
        candidates.append((m.start(), _norm_letters(m.group(1))))
    for m in _ANSWER_LINE_MULTI_RE.finditer(tail):
        if _valid_multi(m.group(1), options):
            candidates.append((m.start(), m.group(1)))
    for m in _EN_ANSWER_RE.finditer(tail):
        candidates.append((m.start(), m.group(1).upper()))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return Extracted("choice", candidates[-1][1])


def extract_text(output: str) -> Extracted | None:
    """开放题:只认 boxed 或「答案:...」整行声明;自由结尾不猜,交给 LLM 层。"""
    tail = _WRAP_RE.sub(r"\1", output[-2000:])
    boxed = _BOXED_RE.findall(tail)
    if boxed:
        return Extracted("text", boxed[-1].strip())
    m = None
    for m in re.finditer(r"(?:最终答案|答案)\s*(?:是|为|:|:)\s*(.+)", tail):
        pass
    if m:
        value = m.group(1).strip().rstrip("。.」】")
        if value:
            return Extracted("text", value)
    return None


def extract(output: str, is_choice: bool, options: Iterable[str] | None = None) -> Extracted | None:
    return extract_choice(output, options) if is_choice else extract_text(output)
