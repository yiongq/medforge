"""弃权验收测试:覆盖率 / 选择性准确率 / 严格准确率 / 弃权精度召回,用手算得出的小卷钉死。"""

from __future__ import annotations

import json

import pytest

from medforge.eval import abstain_report as ar


def _scored(run_dir, name: str, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"{name}.scored.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )


def _ok(i: int) -> dict:
    return {"id": f"q{i}", "correct": True, "method": "rule", "detail": ""}


def _bad(i: int) -> dict:
    return {"id": f"q{i}", "correct": False, "method": "rule", "detail": ""}


def _abstain(i: int) -> dict:
    """模型主动写「答案:不确定」→ 验证器判 abstain + declared。"""
    return {"id": f"q{i}", "correct": None, "method": "abstain", "detail": "declared"}


def _unparsable(i: int) -> dict:
    """验证器抽不出答案:同样是 method=abstain,但 detail 不是 declared——不算主动弃权。"""
    return {"id": f"q{i}", "correct": None, "method": "abstain", "detail": ""}


def _unfinished(i: int) -> dict:
    """撞 max_tokens 没交卷:既不是弃权也不是答错(base-v3-abstain/cmexam 实测 631/2000)。"""
    return {"id": f"q{i}", "correct": None, "method": "unfinished", "detail": ""}


def test_declared_abstain_is_distinguished_from_verifier_abstain():
    assert ar.is_declared_abstain(_abstain(1)) is True
    assert ar.is_declared_abstain(_unparsable(1)) is False
    assert ar.is_declared_abstain({"id": "q", "correct": None, "method": "unfinished"}) is False
    assert ar.is_unusable(_unfinished(1)) is True
    assert ar.is_unusable({"id": "q", "correct": None, "method": "missing"}) is True
    assert ar.is_unusable(_abstain(1)) is False and ar.is_unusable(_bad(1)) is False


def test_unfinished_is_not_counted_as_answered():
    """未收尾必须从覆盖率的分子里扣掉,否则一条三成截断的臂会打印覆盖率 100%,
    再把截断造成的准确率塌陷读成「弃权训练毁了作答能力」。"""
    ref = {f"q{i}": _ok(i) for i in range(10)}
    run = {f"q{i}": (_unfinished(i) if i < 3 else _ok(i)) for i in range(10)}
    s = ar.selective("cmexam", run, ref)
    assert (s.unusable, s.abstained, s.answered) == (3, 0, 7)
    assert s.coverage == 0.7 and s.unusable_rate == 0.3
    assert s.selective_acc == 1.0  # 交了卷的 7 题全对——塌陷全在截断那一列
    assert s.strict_acc == 0.7


def test_selective_metrics_hand_computed():
    """10 题:新 run 弃权 4 题(其中 3 题参照本来就错),答了 6 题对 5 题;参照 10 题对 6 题。"""
    ref = {f"q{i}": (_ok(i) if i < 6 else _bad(i)) for i in range(10)}  # 参照对 q0..q5
    run = {}
    for i in range(10):
        if i in (6, 7, 8, 3):  # 弃权 4 题:q6/q7/q8 参照错,q3 参照对
            run[f"q{i}"] = _abstain(i)
        elif i == 9:
            run[f"q{i}"] = _bad(i)  # 答了但错
        else:
            run[f"q{i}"] = _ok(i)  # q0,q1,q2,q4,q5 答对
    s = ar.selective("cmexam", run, ref)
    assert (s.n, s.abstained, s.correct, s.ref_correct, s.hit, s.ref_wrong) == (10, 4, 5, 6, 3, 4)
    assert (s.unusable, s.answered) == (0, 6)
    assert s.coverage == 0.6  # 答了 6 / 10
    assert s.abstain_rate == 0.4
    assert abs(s.selective_acc - 5 / 6) < 1e-9  # 答的 6 题里对 5 题
    assert s.strict_acc == 0.5  # 弃权计错:5 / 10
    assert s.ref_acc == 0.6
    assert s.precision == 0.75  # 弃权的 4 题里 3 题参照本来就错
    assert s.recall == 0.75  # 参照错的 4 题里弃掉了 3 题


def test_only_common_ids_are_paired():
    """抽样卷 / 断点续跑都可能不齐:分母只取共同题目。"""
    ref = {"q1": _ok(1), "q2": _bad(2)}
    run = {"q1": _abstain(1), "q3": _abstain(3)}
    s = ar.selective("cmb-val", run, ref)
    assert s.n == 1 and s.abstained == 1 and s.hit == 0 and s.recall == 0.0


def test_zero_division_guards():
    s = ar.selective("empty", {}, {})
    assert (s.n, s.coverage, s.selective_acc, s.precision, s.recall) == (0, 0.0, 0.0, 0.0, 0.0)
    # 一题都没弃权:精度分母为 0,不能炸
    s2 = ar.selective("x", {"q1": _ok(1)}, {"q1": _ok(1)})
    assert s2.precision == 0.0 and s2.recall == 0.0 and s2.coverage == 1.0


def test_cli_writes_markdown_table(tmp_path, capsys):
    """配对的两条臂必须同提示词同解码、只差权重,所以 --run 用的是 sample 臂而不是 abstain 臂。"""
    runs = tmp_path / "runs"
    _scored(runs / "abstain-v3-sample", "cmexam", [_abstain(0), _ok(1), _ok(2), _bad(3), _unfinished(4)])
    _scored(runs / "distill-v3-sample", "cmexam", [_bad(0), _ok(1), _ok(2), _ok(3), _ok(4)])
    out = tmp_path / "abstain-selective.md"
    ar.main(
        [
            "--run",
            "abstain-v3-sample",
            "--ref",
            "distill-v3-sample",
            "--sets",
            "cmexam,cmb-val",
            "--runs-dir",
            str(runs),
            "--out",
            str(out),
        ]
    )
    text = out.read_text(encoding="utf-8")
    # 5 题:1 弃权 + 1 未收尾 → 覆盖率 3/5;交卷的 3 题里对 2 题;严格 2/5;参照对 4/5
    assert "| cmexam | 5 | 60.0% | 20.0% | 20.0% | 66.7% | 40.0% | 80.0% | 100.0% | 100.0% |" in text
    assert "未收尾/缺失" in text
    assert "cmb-val" not in text  # 缺 scored 的卷跳过而不是报 0
    assert "跳过 cmb-val" in capsys.readouterr().out


def test_cli_exits_when_nothing_pairs(tmp_path):
    with pytest.raises(SystemExit):
        ar.main(["--run", "a", "--ref", "b", "--sets", "cmexam", "--runs-dir", str(tmp_path)])
