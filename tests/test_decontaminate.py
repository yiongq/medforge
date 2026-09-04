"""去污染层单测:撞题/改写/无关三种关系 + [review] 爆炸与短文本回归锁。"""

from medforge.data.decontaminate import (
    EXACT_ONLY,
    NGRAM,
    NGRAM_ONLY,
    TOP_K_HITS,
    contaminated_train_ids,
    normalize_text,
    options_text,
    removal_policy,
    removed_train_ids,
    scan,
    scan_exact,
    shingles,
    short_stem_ids,
    unscannable,
)

Q = "患者男性65岁,突发胸骨后压榨性疼痛3小时,心电图示V1-V4导联ST段抬高,最可能的诊断是急性前壁心肌梗死"


def test_identical_is_contaminated():
    hits = scan([("t1", Q)], [("e1", Q)])
    assert hits and hits[0].level == "contaminated"
    assert contaminated_train_ids(hits) == {"t1"}


def test_punctuation_variant_still_hits():
    variant = Q.replace(",", ", ").replace("是", "是 ")
    hits = scan([("t1", variant)], [("e1", Q)])
    assert hits and hits[0].level == "contaminated"


def test_unrelated_no_hit():
    other = "青霉素过敏性休克的首选抢救药物是肾上腺素,应立即皮下或肌内注射"
    assert scan([("t1", other)], [("e1", Q)]) == []


def test_partial_overlap_is_suspicious_not_removed():
    partial = Q[: len(Q) // 2] + ",首选的治疗措施是立即行经皮冠状动脉介入治疗"
    hits = scan([("t1", partial)], [("e1", Q)])
    assert hits and hits[0].level == "suspicious"
    assert contaminated_train_ids(hits) == set()


def test_short_text_shingles():
    assert shingles("短文本") == {"短文本"}
    assert shingles("") == set()


# ---- [review] 回归锁 ----


def test_review_boilerplate_does_not_explode():
    # 5000 训练样本共享同一段模板句:文档频率截断应把模板 shingle 剔出索引,
    # 每道评测题的命中数被 top-k 封顶,不再产生 train×eval 全配对
    template = "本题考查内科学基础知识请根据病例描述选出最佳答案"
    train = [(f"t{i}", template + f"第{i}号病例的正文内容各不相同编号{i:04d}") for i in range(5000)]
    evals = [(f"e{j}", template + f"评测卷独有的题干内容{j:03d}") for j in range(50)]
    hits = scan(train, evals)
    assert len(hits) <= TOP_K_HITS * len(evals)
    assert not contaminated_train_ids(hits)  # 只共享模板不共享正文,不应判污染


def test_review_short_eval_is_unscannable_not_matched():
    # 短题干曾走子串通道,真实数据证明是误报工厂(「FDP」命中 "doses o[f DP]T"):
    # 现在短题干不参与匹配,由 unscannable() 单独暴露给报告
    long_train = "For an 11-month-old child who has received two doses of DPT and polio"
    assert scan([("t1", long_train)], [("e1", "FDP")]) == []
    assert unscannable([("e1", "FDP"), ("e2", Q)]) == ["e1"]


# ---- 精确通道(W2):n-gram 单通道在同源选择题上两头都不对,这里锁住修复 ----

OPTS = {"A": "急性心肌梗死", "B": "心绞痛", "C": "主动脉夹层", "D": "肺栓塞"}
SHORT = "甘味的作用特点是"   # 归一化后 8 字符 < NGRAM,n-gram 通道结构性看不见


class TestExactChannels:
    def test_short_stem_invisible_to_ngram_but_caught_by_exact(self):
        # 回归锁:短题干在 scan() 里被跳过(unscannable),精确通道必须照样查得出
        assert len(normalize_text(SHORT)) < NGRAM
        assert scan([("t1", SHORT)], [("e1", SHORT)]) == []
        assert unscannable([("e1", SHORT)]) == ["e1"]

        hits = scan_exact([("t1", SHORT, OPTS)], [("e1", SHORT, OPTS)])
        assert [h.channel for h in hits] == ["stem_options_exact"]
        assert removed_train_ids([], hits, {"t1": "cmexam-train"}) == {"t1"}

    def test_same_stem_different_options_is_suspicious_not_removed(self):
        # 同考点模板、不同选项:实测 104 道仅题干一致里有 92 道属于这类,剔了就是误剔
        other = dict(OPTS, A="主动脉瓣狭窄")
        hits = scan_exact([("t1", SHORT, other)], [("e1", SHORT, OPTS)])
        assert [h.channel for h in hits] == ["stem_exact"]
        assert removed_train_ids([], hits, {"t1": "cmexam-train"}) == set()

    def test_stem_plus_options_identical_is_removed(self):
        hits = scan_exact([("t1", Q, OPTS)], [("e1", Q, OPTS)])
        assert [h.channel for h in hits] == ["stem_options_exact"]
        assert removed_train_ids([], hits, {"t1": "cmexam-train"}) == {"t1"}

    def test_option_order_and_punctuation_do_not_matter(self):
        # 选项按键排序后拼接 + 共用 normalize_text:字典顺序被打乱、标点全半角不同,仍是同题
        shuffled = {"D": OPTS["D"], "B": OPTS["B"], "A": OPTS["A"], "C": OPTS["C"]}
        noisy = {k: f" {v}, " for k, v in OPTS.items()}
        assert options_text(shuffled) == options_text(OPTS)
        hits = scan_exact([("t1", Q + " ", noisy)], [("e1", Q, OPTS)])
        assert [h.channel for h in hits] == ["stem_options_exact"]

    def test_open_ended_source_degrades_to_stem(self):
        # 开放题没有选项:两条精确通道退化成同一条,不该把 stem_exact 也算成「选项不同」
        assert options_text(None) == ""
        hits = scan_exact([("t1", Q, None)], [("e1", Q, None)])
        assert [h.channel for h in hits] == ["stem_options_exact"]

    def test_unrelated_and_empty_stem_no_hit(self):
        assert scan_exact([("t1", Q, OPTS)], [("e1", "完全不同的题干", OPTS)]) == []
        # 空题干不是同题的证据,否则所有空题干互相牵连
        assert scan_exact([("t1", "", OPTS)], [("e1", "  ", OPTS)]) == []

    def test_stem_exact_hits_are_capped_but_dupes_are_not(self):
        # stem_exact 只进报告,按 top-k 截断防爆炸;stem_options_exact 是剔除判据,一条都不许丢
        train = [(f"t{i}", SHORT, dict(OPTS, A=f"选项{i}")) for i in range(TOP_K_HITS * 3)]
        train += [(f"d{i}", SHORT, OPTS) for i in range(TOP_K_HITS * 3)]
        hits = scan_exact(train, [("e1", SHORT, OPTS)])
        assert sum(1 for h in hits if h.channel == "stem_exact") == TOP_K_HITS
        assert sum(1 for h in hits if h.channel == "stem_options_exact") == TOP_K_HITS * 3

    def test_short_stem_ids_uses_normalized_length(self):
        assert short_stem_ids([("e1", SHORT), ("e2", Q)], 30) == ["e1"]


class TestRemovalPolicy:
    def test_cmexam_train_uses_exact_only(self):
        assert removal_policy("cmexam-train") == EXACT_ONLY
        assert not EXACT_ONLY.ngram and EXACT_ONLY.stem_options_exact and not EXACT_ONLY.stem_exact

    def test_unregistered_source_keeps_ngram(self):
        # 新增训练源忘配 policy 时必须退回 n-gram 单通道,不能静默换判据
        for name in ("med-o1-verifiable", "med-o1-sft-zh", "med-r1-zh", "brand-new-source", ""):
            assert removal_policy(name) == NGRAM_ONLY

    def test_ngram_contamination_removed_only_for_ngram_sources(self):
        # 同一次 build 混跑两类源:各按自己的判据判,不互相污染
        hits = scan([("t1", Q), ("t2", Q)], [("e1", Q)])
        assert {h.train_id for h in hits} == {"t1", "t2"}
        removed = removed_train_ids(hits, [], {"t1": "med-r1-zh", "t2": "cmexam-train"})
        assert removed == {"t1"}          # 开放题源:n-gram 污染档照剔(W1 行为不变)
        assert contaminated_train_ids(hits) == {"t1", "t2"}   # 老 API 语义不变

    def test_exact_hits_ignored_for_open_ended_sources(self):
        # 开放题三源的判据一个字节不改:精确通道对它们只算不剔
        hits = scan_exact([("t1", Q, None)], [("e1", Q, None)])
        assert removed_train_ids([], hits, {"t1": "med-o1-sft-zh"}) == set()
