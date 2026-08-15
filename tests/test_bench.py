"""压测口径单测:用假客户端控制时序,验证 TTFT/TPOT/吞吐的算法而不触网。

压测代码最容易出的错是「口径错但数字好看」——TPOT 少除一个 token、
失败请求混进分位数,都会让结论失真却看不出异常,所以这些算式必须锁住。
"""

from __future__ import annotations

import asyncio
import time

from medforge.serve.bench import ReqResult, one_request, run_level


class _Delta:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str | None) -> None:
        self.delta = _Delta(content)


class _Chunk:
    def __init__(self, content: str | None) -> None:
        self.choices = [_Choice(content)]


class _FakeStream:
    """首 token 延迟 first_ms,之后每 token 间隔 gap_ms。"""

    def __init__(self, n_tokens: int, first_ms: float, gap_ms: float) -> None:
        self.n, self.first, self.gap = n_tokens, first_ms, gap_ms

    def __aiter__(self):
        async def gen():
            await asyncio.sleep(self.first / 1000)
            yield _Chunk("首")
            for _ in range(self.n - 1):
                await asyncio.sleep(self.gap / 1000)
                yield _Chunk("字")
        return gen()


class _FakeCompletions:
    def __init__(self, n_tokens=5, first_ms=40, gap_ms=10, fail=False) -> None:
        self.kw = {"n_tokens": n_tokens, "first_ms": first_ms, "gap_ms": gap_ms}
        self.fail = fail
        self.seen: list[dict] = []

    async def create(self, **kwargs):
        self.seen.append(kwargs)
        if self.fail:
            raise RuntimeError("upstream boom")
        return _FakeStream(**self.kw)


class _FakeClient:
    def __init__(self, **kw) -> None:
        self.chat = type("C", (), {"completions": _FakeCompletions(**kw)})()


def test_tpot_excludes_first_token():
    # 5 token、首 token 100ms、总 200ms → TPOT = (200-100)/(5-1) = 25ms
    r = ReqResult(ttft_ms=100, total_ms=200, out_tokens=5, ok=True)
    assert r.tpot_ms == 25

    # 单 token 不能除零
    assert ReqResult(50, 50, 1, True).tpot_ms == 0


def test_one_request_measures_ttft_and_tokens():
    c = _FakeClient(n_tokens=6, first_ms=60, gap_ms=5)
    r = asyncio.run(one_request(c, "m", "题", 64))
    assert r.ok and r.out_tokens == 6
    assert 45 <= r.ttft_ms <= 140          # 首 token 延迟量级正确(留足调度抖动)
    assert r.total_ms >= r.ttft_ms
    assert c.chat.completions.seen[0]["extra_body"] == {"ignore_eos": True}  # 固定输出长度口径


def test_failure_counted_not_averaged():
    c = _FakeClient(fail=True)
    lv = asyncio.run(run_level(c, "m", ["题"], conc=2, max_tokens=32))
    assert lv["requests"] == 0 and lv["failed"] == 12   # 全失败:不产出分位数,只报失败数
    assert "ttft_p50" not in lv


def test_run_level_aggregates():
    c = _FakeClient(n_tokens=4, first_ms=20, gap_ms=5)
    t0 = time.perf_counter()
    lv = asyncio.run(run_level(c, "m", ["题A", "题B"], conc=4, max_tokens=32))
    wall = time.perf_counter() - t0
    assert lv["concurrency"] == 4
    assert lv["requests"] == 12 and lv["failed"] == 0   # n_req = max(conc*3, 12)
    assert lv["output_tok_s"] > 0 and lv["req_per_s"] > 0
    assert lv["ttft_p50"] <= lv["ttft_p95"]
    assert wall < 5                                      # 并发生效:12 条不是串行等待
