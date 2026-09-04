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
- 短题干曾尝试子串包含通道,真实数据实测是误报工厂:考题「FDP」(3 字符)
  命中英文训练题 "doses o[f DP]T" 的跨词碎片,单源就误剔上千条。
  结论:归一化后 <NGRAM 字符的题干(CMExam 占 12%,如「甘味的作用特点是」)
  题意都在选项里,对开放题训练池无字面泄漏面——正确处理是标「不可扫描」
  进报告(unscannable()),而不是硬匹配。

W2 追加:训练源换成与评测卷同源的 CMExam 官方训练集后,n-gram 单通道两头都不对
(误剔 88% 好数据、又漏掉 12% 短题干里的真重题)。本文件下半部分补了两条**精确**
通道 stem_exact / stem_options_exact,并把「哪条通道算剔除判据」做成按训练源可配
的 REMOVAL_POLICIES —— 开放题三源的原有行为一个字节不改。依据数字见下方注释。
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
        if len(t) < n:
            continue  # 不可扫描:由 unscannable() 单独暴露,调用方写进报告
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


def unscannable(eval_items: list[tuple[str, str]], n: int = NGRAM) -> list[str]:
    """归一化后短于 n 的评测题 id:字面查重对它们无能为力,报告里必须如实公布。"""
    return [eid for eid, text in eval_items if len(normalize_text(text)) < n]


# =====================================================================
# 精确通道(W2 新增):补 n-gram 通道在「同源选择题」上的两处结构性失效
# =====================================================================
#
# 实测依据(cmexam-train 54,497 题 × 主力卷 = CMExam test 固定种子 seed=42 抽 2000 题):
#   - n-gram 通道:污染档 181 题(9.05%)、存疑档 194 题(9.70%)
#   - 真重题(题干+选项完全一致且 gold 相同)只有约 12 道(0.6%)
#   - 仅题干一致 104 道,其中 92 道选项不同 —— 是「同一考点模板、不同选项」的不同题
#   => 照搬 n-gram 阈值当剔除判据会误剔约 88% 的好数据。
#   - 反向漏报:CMExam test 有 835/6810(12.3%)题干归一化后 <NGRAM 字符,scan() 直接
#     跳过,结构性看不见;其中 82 道在 cmexam-train 里有完全相同的题干。
#   => 结论:同源选择题改用精确通道做剔除判据,n-gram 通道只保留召回/报告用途。
#
# 三条通道各自计数、各自可复算,互不覆盖:
#   ngram              字面近似,阈值 CONTAMINATED / SUSPICIOUS(见上半文件)
#   stem_exact         归一化题干精确相等但选项不同 → 同模板不同题,只报存疑不剔
#   stem_options_exact 归一化(题干 + 按字母序拼接的选项)精确相等 → 真重题,剔除
# 精确通道不受 scan() 的「归一化后 <NGRAM 字符跳过」限制:短题干正是它要补的那一块。


def options_text(options: dict[str, str] | None) -> str:
    """选项按字母序(键序)拼接文本;开放题无选项返回空串。

    拼键序而不是原始出现序:同一道题在不同数据源里选项顺序可能被打乱,
    按键排序后同题必得同一串。值参与拼接、键不参与——A/B 互换但内容相同的
    两份拷贝应当算同题(gold 字母是否随之改写由数据源保证,不在字面层管)。
    """
    if not options:
        return ""
    return "".join(v for _, v in sorted(options.items()))


def stem_key(question: str) -> str:
    """题干指纹:沿用 n-gram 通道同一套 _norm 归一化,保证两条通道口径一致。"""
    return normalize_text(question)


def stem_options_key(question: str, options: dict[str, str] | None) -> str:
    """题干+选项指纹。开放题(options 为 None)退化成 stem_key。"""
    return normalize_text(question + options_text(options))


@dataclass
class ExactHit:
    eval_id: str
    train_id: str
    channel: str        # "stem_exact"(题干同、选项不同) | "stem_options_exact"(全同)


# (id, question, options)
ExactItem = tuple[str, str, dict[str, str] | None]


def scan_exact(
    train_items: list[ExactItem],
    eval_items: list[ExactItem],
    top_k: int = TOP_K_HITS,
) -> list[ExactHit]:
    """精确查重:题干哈希桶一次扫完,同时产出 stem_exact 与 stem_options_exact。

    stem_options_exact 命中**不截断**——它是剔除判据,截断就等于静默漏剔;
    stem_exact 命中只进报告,按 top_k 截断(高频模板题干可以命中上百条训练样本)。
    """
    stem_index: dict[str, list[tuple[str, str]]] = defaultdict(list)  # 题干指纹 -> [(train_id, 题干+选项指纹)]
    for tid, q, opts in train_items:
        k = stem_key(q)
        if not k:
            continue  # 空题干不是同题的证据,只会把所有空题干互相牵连
        stem_index[k].append((tid, stem_options_key(q, opts)))

    hits: list[ExactHit] = []
    for eid, q, opts in eval_items:
        k = stem_key(q)
        if not k:
            continue
        cands = stem_index.get(k)
        if not cands:
            continue
        ek = stem_options_key(q, opts)
        same = [ExactHit(eid, tid, "stem_options_exact") for tid, tk in cands if tk == ek]
        diff = [ExactHit(eid, tid, "stem_exact") for tid, tk in cands if tk != ek]
        hits.extend(same)
        hits.extend(diff[:top_k])
    return hits


@dataclass(frozen=True)
class RemovalPolicy:
    """某个训练源的剔除判据:哪几条通道的命中才真的把训练样本扔掉。"""

    ngram: bool = True                  # n-gram 污染档(ratio ≥ CONTAMINATED)
    stem_options_exact: bool = False    # 题干+选项全等
    stem_exact: bool = False            # 仅题干相等(默认永远不剔:同模板不同题占 92/104)


# n-gram 单通道:W1 三个开放题训练源的原有行为,一个字节都不改
NGRAM_ONLY = RemovalPolicy(ngram=True)
# 精确单通道:同源选择题用。n-gram 仍会算、仍进报告,但不作为剔除判据
EXACT_ONLY = RemovalPolicy(ngram=False, stem_options_exact=True)

REMOVAL_POLICIES: dict[str, RemovalPolicy] = {
    # cmexam-train 与主力评测卷同源同分布:n-gram 阈值在这里误剔 88%(见上方实测),
    # 只有「题干+选项全等」才是真重题。短题干真重题由 stem_options_exact 兜住。
    "cmexam-train": EXACT_ONLY,
}


def removal_policy(source: str) -> RemovalPolicy:
    """未登记的训练源一律沿用 n-gram 单通道 —— 新增源不会因为忘配而静默换判据。"""
    return REMOVAL_POLICIES.get(source, NGRAM_ONLY)


def removed_train_ids(
    hits: list[Hit],
    exact_hits: list[ExactHit],
    train_source: dict[str, str],
) -> set[str]:
    """按每条训练样本所属源的判据决定剔除(train_source: train_id -> source 名)。

    同一次 build 里混跑多个源时,每条样本各按自己源的 policy 判——
    不会因为混进一个 cmexam-train 就把开放题源的判据也一起改掉。
    """
    out: set[str] = set()
    for h in hits:
        if h.level == "contaminated" and removal_policy(train_source.get(h.train_id, "")).ngram:
            out.add(h.train_id)
    for e in exact_hits:
        p = removal_policy(train_source.get(e.train_id, ""))
        if (e.channel == "stem_options_exact" and p.stem_options_exact) or (
            e.channel == "stem_exact" and p.stem_exact
        ):
            out.add(e.train_id)
    return out


def short_stem_ids(eval_items: list[tuple[str, str]], limit: int) -> list[str]:
    """归一化后短于 limit 字符的评测题 id:字面近似查重对它们基本无效,报告里要公布占比。"""
    return [eid for eid, text in eval_items if len(normalize_text(text)) < limit]
