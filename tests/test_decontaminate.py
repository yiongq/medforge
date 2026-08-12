"""去污染层单测:撞题/改写/无关三种关系 + [review] 爆炸与短文本回归锁。"""

from medforge.data.decontaminate import (
    TOP_K_HITS,
    contaminated_train_ids,
    scan,
    shingles,
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
