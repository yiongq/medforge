"""弃权数据构造测试:自采样(合成)→ known/unknown/unstable 分类 → 配比 → 教材 + 报告。

样本形状照抄 GPU 上真实落盘的 abstain_samples.jsonl:reasoning 恒为空,思考与作答都在 answer 里,
靠 </think> 分段——所以这里的每条用例都必须自己拼出 </think>,否则测的就不是真实输入。
"""

from __future__ import annotations

import json

import pytest

from medforge.data import build_abstain as ba
from medforge.data.schema import Sample
from medforge.eval.run import PROMPT_CHOICE

THINK = "先逐项分析选项的临床意义,再排除干扰项。" * 12   # 240 字符 ≈ 150 token
RIGHT = "综合以上分析。\n答案:B"
WRONG = "综合以上分析。\n答案:A"
GARBAGE = "综上,这几个选项都有道理,请自行判断。"   # 抽不出答案 → 未声明


def _sample(qid: str, gold: str = "B") -> Sample:
    return Sample(
        id=qid, source="cmexam-train", question=f"题号{qid}:下列哪项正确?",
        gold=gold, options={"A": "甲", "B": "乙", "C": "丙"},
    )


def _row(qid: str, k: int = 0, *, think: str = THINK, final: str = RIGHT,
         finish: str = "stop", tokens: int | None = None, close_think: bool = True) -> dict:
    """一条自采样结果。close_think=False 模拟撞上限时连 </think> 都没写出来的答卷。"""
    answer = f"{think}\n</think>\n\n{final}" if close_think else f"{think}\n{final}"
    return {"id": qid, "k": k, "reasoning": "", "answer": answer,
            "finish_reason": finish, "completion_tokens": tokens}


def _write_pool(tmp_path, samples):
    pool = tmp_path / "abstain_pool.jsonl"
    with pool.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")
    return pool


def _write_samples(tmp_path, rows):
    f = tmp_path / "abstain_samples.jsonl"
    with f.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return f


def _build(tmp_path, samples, rows, **kw):
    """跑一遍 build,默认不混通用回放(需要 data/raw,CI 没有)。返回 (stats, 产出行, 报告文本)。"""
    pool = _write_pool(tmp_path, samples)
    sfile = _write_samples(tmp_path, rows)
    out, report = tmp_path / "sft.jsonl", tmp_path / "report.md"
    kw.setdefault("general_ratio", 0.0)
    stats = ba.build_dataset(pool, sfile, out, report, **kw)
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    return stats, lines, report.read_text(encoding="utf-8")


def _assistant(line: dict) -> str:
    return line["messages"][1]["content"]


# ------------------------------------------------------------------ 分类(纯函数)


def _flags(finished: int, declared: int, correct: int, total: int) -> list[dict]:
    out = []
    for i in range(total):
        out.append({"finished": i < finished, "declared": i < declared, "correct": i < correct})
    return out


def test_classify_thresholds():
    """known = K 次全对;unknown = 0 次对 + 至多 1 条没写完 + 至少 1 条既收尾又声明;其余 unstable。"""
    assert ba.classify(_flags(4, 4, 4, 4)) == ("known", "")
    assert ba.classify(_flags(4, 4, 0, 4)) == ("unknown", "")
    assert ba.classify(_flags(3, 3, 0, 4)) == ("unknown", "")           # 1 条截断:仍算不会
    assert ba.classify(_flags(4, 4, 2, 4)) == ("unstable", "u_partial")  # 半对半错
    assert ba.classify(_flags(2, 2, 0, 4)) == ("unstable", "u_unfinished")  # 2 条截断:是没写完
    assert ba.classify(_flags(4, 0, 0, 4)) == ("unstable", "u_no_declared")  # 全是格式垃圾
    assert ba.classify(_flags(1, 1, 1, 1), min_samples=2) == ("unstable", "u_too_few")
    # 阈值都是 CLI 旋钮:放宽 known 到 3/4、允许 unknown 有 1 条蒙对
    assert ba.classify(_flags(4, 4, 3, 4), known_ratio=0.75) == ("known", "")
    assert ba.classify(_flags(4, 4, 1, 4), unknown_max_correct=1) == ("unknown", "")


def test_downsample_hits_target_ratio_deterministically():
    known, unknown = [f"k{i}" for i in range(20)], [f"u{i}" for i in range(20)]
    kk, uu = ba.downsample(known, unknown, 0.35, seed=42)
    assert len(kk) == 20 and len(uu) == 11 and abs(11 / 31 - 0.35) < 0.01
    assert (kk, uu) == ba.downsample(known, unknown, 0.35, seed=42)   # 固定 seed 可复现
    # 少的一边是 unknown 时改砍 known
    kk2, uu2 = ba.downsample(known, unknown[:2], 0.35, seed=42)
    assert len(uu2) == 2 and len(kk2) == 4


# ------------------------------------------------------------------ 三类样本的成品形状


def test_known_row_is_a_normal_sft_row(tmp_path):
    """K 次全对 → 保留一条正解,assistant 是完整的 <think>…</think> 对且以「答案:B」收尾。"""
    s = _sample("q1")
    rows = [_row("q1", k) for k in range(4)]
    stats, lines, _ = _build(tmp_path, [s], rows)
    assert stats["classes"] == {"known": 1, "unknown": 0, "unstable": 0}
    a = _assistant(lines[0])
    assert a.startswith("<think>\n") and a.count("<think>") == 1 and a.count("</think>") == 1
    assert a.endswith("\n</think>\n\n综合以上分析。\n答案:B")
    assert ba.BRIDGE_SENTENCE not in a


def test_unknown_row_ends_with_declared_abstain(tmp_path):
    """0 次对 → 用模型自己的思考 + 一句过渡,末行恰好是「答案:不确定」。"""
    s = _sample("q1")
    rows = [_row("q1", k, final=WRONG) for k in range(4)]
    stats, lines, _ = _build(tmp_path, [s], rows)
    assert stats["classes"] == {"known": 0, "unknown": 1, "unstable": 0}
    a = _assistant(lines[0])
    assert a.count("<think>") == 1 and a.count("</think>") == 1
    assert a.startswith("<think>\n先逐项分析")            # 思考是模型自己的,不是新编的
    assert f"\n\n{ba.BRIDGE_SENTENCE}\n</think>\n\n" in a  # 过渡句在 think 段内,结论在外
    assert a.endswith("答案:不确定") and "答案:A" not in a
    # 弃权行必须能被验证器认成「主动弃权」而不是「抽不出」
    ext = ba.extract(a.split("</think>")[-1], True, options=s.options)
    assert ext is not None and ext.kind == "abstain"


def test_unstable_partial_is_dropped(tmp_path):
    """半对半错的题一条都不进教材:模型本来就有一半机会答对,教它拒答是净损失。"""
    rows = [_row("q1", 0), _row("q1", 1), _row("q1", 2, final=WRONG), _row("q1", 3, final=WRONG)]
    stats, lines, _ = _build(tmp_path, [_sample("q1")], rows)
    assert stats["classes"]["unstable"] == 1 and stats["counts"]["u_partial"] == 1
    assert lines == [] and stats["n_med"] == 0


def test_truncated_and_undeclared_are_not_unknown(tmp_path):
    """截断 / 抽不出答案的答卷是「没写完」,不是「不会」——不能标成 unknown。"""
    rows = (
        # q1:全错但两条撞上限 → u_unfinished
        [_row("q1", 0, final=WRONG), _row("q1", 1, final=WRONG),
         _row("q1", 2, final=WRONG, finish="length"), _row("q1", 3, final=WRONG, finish="length")]
        # q2:四条都没写出 </think>(finish_reason=stop 也不算收尾)→ u_unfinished
        + [_row("q2", k, final=WRONG, close_think=False) for k in range(4)]
        # q3:写完了但一条都抽不出答案 → u_no_declared
        + [_row("q3", k, final=GARBAGE) for k in range(4)]
        # q4:全错、只有一条截断 → 这才是 unknown
        + [_row("q4", 0, final=WRONG), _row("q4", 1, final=WRONG),
           _row("q4", 2, final=WRONG), _row("q4", 3, final=WRONG, finish="length")]
    )
    stats, lines, _ = _build(tmp_path, [_sample(f"q{i}") for i in (1, 2, 3, 4)], rows)
    c = stats["counts"]
    assert (c["u_unfinished"], c["u_no_declared"]) == (2, 1)
    assert c["unfinished"] == 2 + 4 + 1 and c["undeclared"] == 4
    assert stats["classes"] == {"known": 0, "unknown": 1, "unstable": 3}
    assert len(lines) == 1 and _assistant(lines[0]).endswith("答案:不确定")


def test_pool_missing_ids_are_counted_not_crashed(tmp_path):
    stats, lines, _ = _build(tmp_path, [_sample("q1")], [_row("q1", k) for k in range(4)] + [_row("qX", 0)])
    assert stats["counts"]["pool_missing"] == 1 and len(lines) == 1


# ------------------------------------------------------------------ 提示词同一性 / 选样 / 配比


def test_prompt_is_identical_to_eval_prompt(tmp_path):
    """训练 user 必须与评测提示词逐字相同(8 月 SFT「训练裸题干 / 评测带格式指令」的坑),
    known 与 unknown 两类都要。"""
    s1, s2 = _sample("q1"), _sample("q2")
    rows = [_row("q1", k) for k in range(4)] + [_row("q2", k, final=WRONG) for k in range(4)]
    _, lines, _ = _build(tmp_path, [s1, s2], rows, abstain_ratio=0.5)
    by_user = {ln["messages"][0]["content"]: _assistant(ln) for ln in lines}
    for s in (s1, s2):
        expected = PROMPT_CHOICE.format(question=s.render_question())
        assert expected in by_user
        assert "答案:X" in expected and s.render_question() in expected
    assert len(lines) == 2


def test_pick_takes_median_not_shortest(tmp_path):
    """每题保留一条,按思考长度取中位——最短的错解常是「没想就答」,不能拿来当弃权范例。"""
    lens = {0: 10, 1: 20, 2: 40}
    rows = [_row("q1", k, think="中位测试用的中文思考。" * n, final=WRONG) for k, n in lens.items()]
    _, lines, _ = _build(tmp_path, [_sample("q1")], rows)
    a = _assistant(lines[0])
    assert "中位测试用的中文思考。" * 20 in a
    assert "中位测试用的中文思考。" * 21 not in a   # 既不是最长也不是最短


def test_abstain_ratio_controls_share(tmp_path):
    """--abstain-ratio 控制弃权行占医疗行的比例,下采样多的一边。"""
    samples = [_sample(f"k{i}") for i in range(20)] + [_sample(f"u{i}") for i in range(20)]
    rows = ([_row(f"k{i}", k) for i in range(20) for k in range(2)]
            + [_row(f"u{i}", k, final=WRONG) for i in range(20) for k in range(2)])
    stats, lines, report = _build(tmp_path, samples, rows, abstain_ratio=0.35)
    assert (stats["n_known_rows"], stats["n_unknown_rows"]) == (20, 11)
    assert abs(stats["abstain_share"] - 0.35) < 0.02 and len(lines) == 31
    assert sum(_assistant(ln).endswith("答案:不确定") for ln in lines) == 11
    assert "占医疗行" in report

    # 同一份输入换比例:0.5 时两边打平
    half, _, _ = _build(tmp_path, samples, rows, abstain_ratio=0.5)
    assert (half["n_known_rows"], half["n_unknown_rows"]) == (20, 20)


def test_too_long_rows_are_dropped_in_data_layer(tmp_path):
    """总长超训练 max_length 的在数据侧剔除,不交给框架 truncation(它截掉的正是末行的答案)。"""
    stats, lines, _ = _build(tmp_path, [_sample("q1")], [_row("q1", k) for k in range(2)],
                             train_max_length=50)
    assert stats["counts"]["p_too_long"] == 1 and stats["counts"]["dropped_at_pick"] == 1
    assert lines == []


def test_known_row_requires_answer_literal(tmp_path):
    """known 行的作答段必须含「答案:」字面,与评测提示词要求的格式逐字一致。"""
    rows = [_row("q1", k, final="综上,\\boxed{B}") for k in range(2)]
    stats, lines, _ = _build(tmp_path, [_sample("q1")], rows)
    assert stats["classes"]["known"] == 1 and stats["counts"]["p_no_literal"] == 1 and lines == []


def test_general_replay_mix(tmp_path):
    """--general-ratio 复用 build_sft.mix 的公式:通用条数 = 医疗 × r/(1-r),整体打散。"""
    samples = [_sample(f"k{i}") for i in range(2)] + [_sample(f"u{i}") for i in range(2)]
    rows = ([_row(f"k{i}", 0) for i in range(2)] + [_row(f"u{i}", 0, final=WRONG) for i in range(2)])
    general = [{"messages": [{"role": "user", "content": f"通用{i}"},
                             {"role": "assistant", "content": "好"}]} for i in range(50)]
    stats, lines, report = _build(tmp_path, samples, rows, abstain_ratio=0.5,
                                  general_ratio=0.5, general_loader=lambda: general)
    assert (stats["n_med"], stats["n_general"], stats["n_total"]) == (4, 4, 8) and len(lines) == 8
    assert "医疗 4 行 + 通用回放 4 行 = 8 行" in report


# ------------------------------------------------------------------ 报告 / CLI


def test_report_matches_dataset(tmp_path):
    samples = [_sample(f"k{i}") for i in range(4)] + [_sample("u1"), _sample("p1")]
    rows = ([_row(f"k{i}", k) for i in range(4) for k in range(2)]
            + [_row("u1", k, final=WRONG) for k in range(2)]
            + [_row("p1", 0), _row("p1", 1, final=WRONG)])       # 半对半错 → unstable
    stats, lines, report = _build(tmp_path, samples, rows, abstain_ratio=0.5)
    assert stats["n_rows"] == 12 and stats["n_questions"] == 6
    assert stats["classes"] == {"known": 4, "unknown": 1, "unstable": 1}
    assert "| known | 4 | 66.7% |" in report and "| unknown | 1 | 16.7% |" in report
    assert f"医疗 {stats['n_med']} 行" in report and len(lines) == stats["n_med"]
    assert "收尾率 100.0%" in report and "声明率 100.0%" in report
    assert "p50" in report and "BRIDGE_SENTENCE" in report and "abstain_report" in report


def test_cli_end_to_end(tmp_path):
    samples = [_sample("q1"), _sample("q2")]
    pool, sfile = _write_pool(tmp_path, samples), _write_samples(
        tmp_path, [_row("q1", k) for k in range(2)] + [_row("q2", k, final=WRONG) for k in range(2)]
    )
    out, report = tmp_path / "sft_abstain_v1.jsonl", tmp_path / "abstain-dataset.md"
    ba.main(["--samples", str(sfile), "--pool", str(pool), "--out", str(out),
             "--report", str(report), "--general-ratio", "0", "--abstain-ratio", "0.5"])
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and report.exists()
    assert sum(_assistant(ln).endswith("答案:不确定") for ln in lines) == 1


def test_cli_exits_when_no_abstain_rows(tmp_path):
    """一条弃权样本都没有的教材教不出弃权:必须以非零退出码喊停,而不是安静地产出一份普通 SFT 文件。"""
    pool = _write_pool(tmp_path, [_sample("q1")])
    sfile = _write_samples(tmp_path, [_row("q1", k) for k in range(2)])
    with pytest.raises(SystemExit):
        ba.main(["--samples", str(sfile), "--pool", str(pool), "--out", str(tmp_path / "o.jsonl"),
                 "--report", str(tmp_path / "r.md"), "--general-ratio", "0"])
