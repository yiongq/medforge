"""训练集 vs 评测集 去污染(decontamination)——字面层。

为什么这是硬步骤:医疗考试题库长期公开,训练数据和评测集撞题会把
「背过答案」伪装成「能力提升」,整个项目的涨分数字就全部作废。
业界做法是 n-gram 字面查重 + 语义查重双层(GPT-3 用 13-gram token,
Foundation-Sec-8B 用 8-gram);本文件是字面层,语义层(embedding)
在 W1b 随 build 管线 CLI 一起接入——当前 README/报告口径也如实只写字面层。

中文没有天然分词,这里用【暂定】字符级 10-gram:约等于英文 8-gram token
的信息量;阈值先取「评测题 shingle 命中率 ≥ 0.8 判污染、≥ 0.3 报告存疑」,
校准方式=人工抽看被判污染的样本,再调。

W1a 审查修复记录:
- 共享模板句曾导致 train×eval 存疑配对爆炸(合成实测 250 万 Hit)→
  文档频率截断(模板 shingle 剔出索引)+ 每题只留 top-k 命中
- 归一化后不足 10 字的短评测题曾必然漏报 → 走独立的子串包含通道
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass

# 归一化:去掉空白/标点/大小写差异,只留文字内容——
# 否则同一道题因标点全半角不同就查不出来
_STRIP_RE = re.compile(r"[\s\W_]+", re.UNICODE)

NGRAM = 10            # 【暂定】字符级 10-gram
CONTAMINATED = 0.8    # 【暂定】命中率 ≥ 0.8 → 判污染,训练侧剔除
SUSPICIOUS = 0.3      # 【暂定】命中率 ≥ 0.3 → 报告存疑,人工抽看
BOILERPLATE_DF = 0.005  # 【暂定】shingle 出现在 >0.5% 训练样本(下限 20 条)→ 模板噪声,剔出索引
TOP_K_HITS = 20       # 每道评测题最多保留 top-k 命中:人工抽看只看得过来这么多


def normalize_text(text: str) -> str:
    return _STRIP_RE.sub("", text).lower()


def _shingles_norm(t: str, n: int) -> set[str]:
    if len(t) < n:
        return {t} if t else set()
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def shingles(text: str, n: int = NGRAM) -> set[str]:
    return _shingles_norm(normalize_text(text), n)


@dataclass
class Hit:
    eval_id: str
    train_id: str
    ratio: float        # 评测题的 shingle 被该训练样本覆盖的比例
    level: str          # "contaminated" | "suspicious"


def scan(
    train_items: list[tuple[str, str]],
    eval_items: list[tuple[str, str]],
    n: int = NGRAM,
) -> list[Hit]:
    """扫描训练集对评测集的字面污染。

    入参都是 (id, text)。方向刻意是「评测题被训练样本覆盖多少」:
    训练样本比评测题长得多时,反向比例会被稀释而漏报。
    倒排索引把复杂度降到只对共享 shingle 的对子计数;
    高频模板 shingle 先按文档频率剔除,防配对爆炸。
    """
    train_norm = [(tid, normalize_text(text)) for tid, text in train_items]

    index: dict[str, set[str]] = defaultdict(set)  # shingle -> {train_id}
    for tid, t in train_norm:
        for s in _shingles_norm(t, n):
            index[s].add(tid)
    df_cap = max(20, int(len(train_norm) * BOILERPLATE_DF))
    index = {s: tids for s, tids in index.items() if len(tids) <= df_cap}

    hits: list[Hit] = []
    for eid, text in eval_items:
        t = normalize_text(text)
        if not t:
            continue
        if len(t) < n:
            # 短文本通道:10-gram 索引对它天然失明,直接做子串包含(短题极少,线性扫得起)
            for tid, tt in train_norm:
                if t in tt:
                    hits.append(Hit(eid, tid, 1.0, "contaminated"))
            continue
        ss = _shingles_norm(t, n)
        counter: dict[str, int] = defaultdict(int)  # train_id -> 命中 shingle 数
        for s in ss:
            for tid in index.get(s, ()):
                counter[tid] += 1
        item_hits: list[Hit] = []
        for tid, cnt in counter.items():
            ratio = cnt / len(ss)
            if ratio >= CONTAMINATED:
                item_hits.append(Hit(eid, tid, round(ratio, 3), "contaminated"))
            elif ratio >= SUSPICIOUS:
                item_hits.append(Hit(eid, tid, round(ratio, 3), "suspicious"))
        item_hits.sort(key=lambda h: -h.ratio)
        hits.extend(item_hits[:TOP_K_HITS])
    hits.sort(key=lambda h: -h.ratio)
    return hits


def contaminated_train_ids(hits: list[Hit]) -> set[str]:
    """要从训练集剔除的样本 id。存疑档不剔除、只进报告——宁可人工看,不静默扔数据。"""
    return {h.train_id for h in hits if h.level == "contaminated"}
