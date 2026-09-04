"""Claude Code provider 单测:subprocess 全程打桩,绝不真的调 CLI。

这里锁的是三件容易悄悄坏掉的事:
  1. 进程环境必须干净(CLAUDECODE / CLAUDE_* 一个都不能漏),否则 CLI 认为自己被嵌套调用而拒绝;
  2. 任何一种失败(非零退出 / 超时 / subtype != success / 输出不是 JSON)都要变成 ClaudeCodeError;
  3. 判卷员切到这个后端后仍然「永不抛」:失败退化成 Verdict(None, "llm", ...),不炸掉整卷。
"""

from __future__ import annotations

import json
import subprocess

import pytest

from medforge.data.schema import Sample
from medforge.verify import claude_code as cc

SUCCESS = {
    "subtype": "success",
    "result": '{"correct": true, "reason": "结论一致"}',
    "structured_output": {"correct": True, "reason": "结论一致"},
    "usage": {"input_tokens": 12, "cache_read_input_tokens": 31000, "output_tokens": 345},
    "total_cost_usd": 0.0123,
    "num_turns": 1,
}


def fake_cli(
    monkeypatch, payload: dict | str, *, returncode: int = 0, stderr: str = "", boom=None
) -> list[dict]:
    """把 subprocess.run 换成剧本;返回记录了 argv/env/cwd 的调用列表。"""
    calls: list[dict] = []

    def run(argv, **kw):
        calls.append({"argv": argv, "env": kw.get("env"), "cwd": kw.get("cwd"), "timeout": kw.get("timeout")})
        if boom is not None:
            raise boom
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.CompletedProcess(argv, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(cc.subprocess, "run", run)
    return calls


class TestQuery:
    def test_success_parses_text_structured_and_usage(self, monkeypatch):
        fake_cli(monkeypatch, SUCCESS)
        r = cc.claude_code_query("判一下", model="claude-opus-5", system_prompt="你是判卷员")
        assert r.structured == {"correct": True, "reason": "结论一致"}
        assert r.text.startswith("{") and r.output_tokens == 345 and r.cost_usd == 0.0123
        assert r.raw["num_turns"] == 1

    def test_argv_pins_model_headless_flags_and_schema(self, monkeypatch):
        calls = fake_cli(monkeypatch, SUCCESS)
        schema = {"type": "object", "properties": {"correct": {"type": ["boolean", "null"]}}}
        cc.claude_code_query(
            "q", model="claude-sonnet-5", system_prompt="sys", json_schema=schema, effort="high"
        )
        argv = calls[0]["argv"]
        # 每个开关都是必要的:少一个就可能拉起工具/项目配置,判分口径被环境悄悄改写
        for flag in (
            "--output-format",
            "--max-turns",
            "--tools",
            "--setting-sources",
            "--disable-slash-commands",
            "--no-session-persistence",
        ):
            assert flag in argv
        assert argv[argv.index("--model") + 1] == "claude-sonnet-5"
        assert argv[argv.index("--max-turns") + 1] == "1"
        assert argv[argv.index("--system-prompt") + 1] == "sys"
        assert argv[argv.index("--effort") + 1] == "high"
        assert json.loads(argv[argv.index("--json-schema") + 1]) == schema

    def test_no_schema_no_effort_means_no_flags(self, monkeypatch):
        calls = fake_cli(monkeypatch, SUCCESS)
        cc.claude_code_query("q", model="claude-opus-5")
        assert "--json-schema" not in calls[0]["argv"] and "--effort" not in calls[0]["argv"]

    def test_env_drops_nested_session_markers(self, monkeypatch):
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setenv("CLAUDE_CODE_ENTRYPOINT", "cli")
        monkeypatch.setenv("PATH_KEEPER", "keep-me")
        calls = fake_cli(monkeypatch, SUCCESS)
        cc.claude_code_query("q", model="claude-opus-5")
        env = calls[0]["env"]
        assert "CLAUDECODE" not in env and not [k for k in env if k.startswith("CLAUDE_")]
        assert env["PATH_KEEPER"] == "keep-me"  # 只摘嵌套标记,别把整份环境清空

    def test_runs_in_empty_scratch_dir(self, monkeypatch, tmp_path):
        calls = fake_cli(monkeypatch, SUCCESS)
        cc.claude_code_query("q", model="claude-opus-5")
        # 默认临时空目录:不能落在项目里,否则 CLAUDE.md/memory 会被当上下文塞进提示词
        assert calls[0]["cwd"] and "medforge-cc-" in calls[0]["cwd"]
        cc.claude_code_query("q", model="claude-opus-5", cwd=str(tmp_path))
        assert calls[1]["cwd"] == str(tmp_path)

    def test_non_success_subtype_raises(self, monkeypatch):
        fake_cli(monkeypatch, {"subtype": "error_max_turns", "result": "超轮次"})
        with pytest.raises(cc.ClaudeCodeError, match="未成功"):
            cc.claude_code_query("q", model="claude-opus-5")

    def test_nonzero_exit_raises_with_stderr(self, monkeypatch):
        fake_cli(monkeypatch, "", returncode=1, stderr="not logged in")
        with pytest.raises(cc.ClaudeCodeError, match="not logged in"):
            cc.claude_code_query("q", model="claude-opus-5")

    def test_timeout_raises(self, monkeypatch):
        fake_cli(monkeypatch, SUCCESS, boom=subprocess.TimeoutExpired("claude", 5))
        with pytest.raises(cc.ClaudeCodeError, match="超时"):
            cc.claude_code_query("q", model="claude-opus-5", timeout=5)

    def test_bad_json_raises(self, monkeypatch):
        fake_cli(monkeypatch, "Usage: claude [options]")
        with pytest.raises(cc.ClaudeCodeError, match="不是 JSON"):
            cc.claude_code_query("q", model="claude-opus-5")

    def test_missing_cli_raises(self, monkeypatch):
        fake_cli(monkeypatch, SUCCESS, boom=FileNotFoundError("claude"))
        with pytest.raises(cc.ClaudeCodeError, match="无法执行"):
            cc.claude_code_query("q", model="claude-opus-5")

    def test_bad_effort_rejected_before_spawning(self, monkeypatch):
        calls = fake_cli(monkeypatch, SUCCESS)
        with pytest.raises(ValueError, match="effort"):
            cc.claude_code_query("q", model="claude-opus-5", effort="ultra")
        assert calls == []


class TestParseJsonObject:
    def test_scrapes_object_from_prose(self):
        assert cc.parse_json_object('好的:{"correct": false, "reason": "选项不同"} 完毕')["correct"] is False

    def test_returns_none_on_garbage(self):
        assert cc.parse_json_object("没有 JSON") is None and cc.parse_json_object("{不是合法 JSON}") is None


def _sample() -> Sample:
    return Sample(id="t", source="t", question="题干", gold="阿司匹林")


class TestVerifyByLlmProvider:
    """判卷员的 provider 切换;openai 路径的行为不变(默认值仍是 openai)。"""

    def _use_claude_code(self, monkeypatch, model: str = "claude-opus-5") -> None:
        monkeypatch.setattr("medforge.env.load_env", lambda: None)  # 别让本机 .env 混进来
        monkeypatch.setenv("MEDFORGE_JUDGE_PROVIDER", "claude-code")
        monkeypatch.setenv("MEDFORGE_JUDGE_MODEL", model)

    def test_default_provider_is_openai(self, monkeypatch):
        monkeypatch.delenv("MEDFORGE_JUDGE_PROVIDER", raising=False)
        from medforge.verify.verifier import judge_provider

        assert judge_provider() == "openai"

    def test_unknown_provider_falls_back_to_openai(self, monkeypatch):
        monkeypatch.setenv("MEDFORGE_JUDGE_PROVIDER", "gemini-cli")
        from medforge.verify.verifier import judge_provider

        assert judge_provider() == "openai"

    def test_missing_judge_env_depends_on_provider(self, monkeypatch):
        monkeypatch.setattr("medforge.env.load_env", lambda: None)
        from medforge.verify.verifier import missing_judge_env

        monkeypatch.setenv("MEDFORGE_JUDGE_PROVIDER", "claude-code")
        monkeypatch.setenv("MEDFORGE_JUDGE_MODEL", "claude-opus-5")
        monkeypatch.delenv("MEDFORGE_JUDGE_BASE_URL", raising=False)
        monkeypatch.delenv("MEDFORGE_JUDGE_API_KEY", raising=False)
        assert missing_judge_env() == []  # claude-code 不需要 base_url/api_key
        monkeypatch.setenv("MEDFORGE_JUDGE_PROVIDER", "openai")
        assert missing_judge_env() == ["MEDFORGE_JUDGE_BASE_URL", "MEDFORGE_JUDGE_API_KEY"]

    def test_structured_output_becomes_verdict(self, monkeypatch):
        self._use_claude_code(monkeypatch)
        monkeypatch.setenv("MEDFORGE_JUDGE_EFFORT", "medium")
        seen: dict = {}

        def fake_query(prompt, **kw):
            seen.update(kw, prompt=prompt)
            return cc.ClaudeCodeResult(
                text="",
                structured={"correct": True, "reason": "同义"},
                output_tokens=9,
                cost_usd=0.001,
                raw={},
            )

        monkeypatch.setattr(cc, "claude_code_query", fake_query)
        from medforge.verify.verifier import verify_by_llm

        v = verify_by_llm(_sample(), "最终答案:乙酰水杨酸")
        assert (v.correct, v.method, v.detail) == (True, "llm", "同义")
        assert seen["model"] == "claude-opus-5" and seen["effort"] == "medium"
        assert seen["json_schema"]["properties"]["correct"]["type"] == ["boolean", "null"]

    def test_falls_back_to_regex_when_structured_missing(self, monkeypatch):
        self._use_claude_code(monkeypatch)
        monkeypatch.setattr(
            cc,
            "claude_code_query",
            lambda prompt, **kw: cc.ClaudeCodeResult(
                text='判定如下 {"correct": false, "reason": "不是同一种药"}',
                structured=None,
                output_tokens=9,
                cost_usd=None,
                raw={},
            ),
        )
        from medforge.verify.verifier import verify_by_llm

        v = verify_by_llm(_sample(), "最终答案:阿莫西林")
        assert (v.correct, v.method) == (False, "llm")

    def test_unparsable_output_abstains(self, monkeypatch):
        self._use_claude_code(monkeypatch)
        monkeypatch.setattr(
            cc,
            "claude_code_query",
            lambda prompt, **kw: cc.ClaudeCodeResult(
                text="我拿不准", structured=None, output_tokens=1, cost_usd=None, raw={}
            ),
        )
        from medforge.verify.verifier import verify_by_llm

        v = verify_by_llm(_sample(), "最终答案:阿司匹林")
        assert v.correct is None and v.method == "llm"

    def test_cli_failure_never_raises(self, monkeypatch):
        """判分链的契约:LLM 层失败只弃权。scored 文件是 "w" 模式,整卷炸掉会连上一版判分一起丢。"""
        self._use_claude_code(monkeypatch)

        def boom(prompt, **kw):
            raise cc.ClaudeCodeError("claude 退出码 1: not logged in")

        monkeypatch.setattr(cc, "claude_code_query", boom)
        from medforge.verify.verifier import verify, verify_by_llm

        v = verify_by_llm(_sample(), "最终答案:阿司匹林钠")
        assert v.correct is None and v.method == "llm" and "ClaudeCodeError" in v.detail
        # 整条判分链同样不抛(规则层弃权 → LLM 兜底 → 弃权)
        assert verify(_sample(), "最终答案:阿司匹林钠", thinking=False).method == "llm"

    def test_missing_model_abstains_without_calling_cli(self, monkeypatch):
        monkeypatch.setattr("medforge.env.load_env", lambda: None)
        monkeypatch.setenv("MEDFORGE_JUDGE_PROVIDER", "claude-code")
        monkeypatch.delenv("MEDFORGE_JUDGE_MODEL", raising=False)
        calls = fake_cli(monkeypatch, SUCCESS)
        from medforge.verify.verifier import verify_by_llm

        v = verify_by_llm(_sample(), "最终答案:阿司匹林钠")
        assert v.correct is None and v.method == "abstain" and calls == []
