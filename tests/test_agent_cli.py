"""End-to-end CLI tests for the 'supereval agent' subcommand."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from supereval.cli import app
from supereval.agent.models import (
    AgentExpected,
    AgentRunResult,
    AgentTask,
    AgentTestCase,
    MockResponse,
    Step,
    StepType,
    ToolCall,
    Trajectory,
    TrajectoryScore,
    AgentCaseResult,
)

runner = CliRunner()


def invoke(*args, env=None):
    return runner.invoke(app, ["agent"] + list(args), env=env)


@pytest.fixture
def cli_env(datasets_dir, tmp_path):
    return {
        "SUPEREVAL_DATASETS_DIR": str(datasets_dir),
        "SUPEREVAL_DB_PATH": str(tmp_path / "test.db"),
    }


# ---------------------------------------------------------------------------
# dataset create
# ---------------------------------------------------------------------------

class TestAgentDatasetCreate:
    def test_creates_dataset(self, cli_env):
        result = invoke("dataset", "create", "my-agent", env=cli_env)
        assert result.exit_code == 0
        assert "Created" in result.output

    def test_duplicate_fails(self, cli_env):
        invoke("dataset", "create", "my-agent", env=cli_env)
        result = invoke("dataset", "create", "my-agent", env=cli_env)
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_creates_with_tool_specs(self, cli_env, tmp_path):
        tools = [{"name": "search", "description": "Search", "parameters": {}}]
        specs_file = tmp_path / "tools.json"
        specs_file.write_text(json.dumps(tools))
        result = invoke(
            "dataset", "create", "my-agent",
            "--tool-specs", str(specs_file),
            env=cli_env,
        )
        assert result.exit_code == 0

    def test_invalid_tool_specs_fails(self, cli_env, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not json at all")
        result = invoke(
            "dataset", "create", "my-agent",
            "--tool-specs", str(bad_file),
            env=cli_env,
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# dataset list
# ---------------------------------------------------------------------------

class TestAgentDatasetList:
    def test_empty_shows_message(self, cli_env):
        result = invoke("dataset", "list", env=cli_env)
        assert result.exit_code == 0
        assert "No agent datasets" in result.output

    def test_shows_created_dataset(self, cli_env, agent_meta):
        result = invoke("dataset", "list", env=cli_env)
        assert result.exit_code == 0
        assert "test-agent" in result.output


# ---------------------------------------------------------------------------
# dataset show
# ---------------------------------------------------------------------------

class TestAgentDatasetShow:
    def test_shows_metadata(self, cli_env, agent_meta):
        result = invoke("dataset", "show", "test-agent", env=cli_env)
        assert result.exit_code == 0
        assert "test-agent" in result.output
        assert "search" in result.output  # tool name

    def test_shows_cases_preview(self, cli_env, agent_dataset_with_cases):
        result = invoke("dataset", "show", "test-agent", env=cli_env)
        assert result.exit_code == 0
        assert "item-42" in result.output or "us-west-2" in result.output


# ---------------------------------------------------------------------------
# dataset add-cases
# ---------------------------------------------------------------------------

class TestAgentDatasetAddCases:
    def test_adds_valid_cases(self, cli_env, agent_meta, tmp_path):
        case = AgentTestCase(
            input=AgentTask(task="find bucket region"),
            tools={"search": MockResponse(response=["us-east-1"])},
            expected=AgentExpected(answer="us-east-1"),
        )
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text(case.model_dump_json() + "\n")
        result = invoke(
            "dataset", "add-cases", "test-agent", "--from", str(cases_file),
            env=cli_env,
        )
        assert result.exit_code == 0
        assert "Added 1 case" in result.output

    def test_rejects_invalid_json(self, cli_env, agent_meta, tmp_path):
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"not": "a valid case"}\n')
        result = invoke(
            "dataset", "add-cases", "test-agent", "--from", str(bad_file),
            env=cli_env,
        )
        assert result.exit_code != 0
        assert "error" in result.output.lower()

    def test_rejects_missing_file(self, cli_env, agent_meta):
        result = invoke(
            "dataset", "add-cases", "test-agent", "--from", "/nonexistent/path.jsonl",
            env=cli_env,
        )
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# dataset validate
# ---------------------------------------------------------------------------

class TestAgentDatasetValidate:
    def test_valid_dataset(self, cli_env, agent_dataset_with_cases):
        result = invoke("dataset", "validate", "test-agent", env=cli_env)
        assert result.exit_code == 0
        assert "OK" in result.output


# ---------------------------------------------------------------------------
# agent run
# ---------------------------------------------------------------------------

def _make_fake_agent_run_result(dataset="test-agent", run_id="agent_test001"):
    cases = [
        AgentCaseResult(
            case_id="atc_case001",
            description="Find item region",
            vars={"task": "Which region is item-42 in?"},
            trajectory=Trajectory(
                steps=[
                    Step(type=StepType.tool_call, content="",
                         tool_call=ToolCall(name="search", arguments={}, result=["r"])),
                    Step(type=StepType.answer, content="us-west-2"),
                ],
                final_answer="us-west-2",
            ),
            score=TrajectoryScore(
                passed=True,
                composite_score=1.0,
                answer_score=1.0,
                tool_score=1.0,
                efficiency_score=1.0,
                step_count=2,
                tool_call_count=1,
            ),
            latency_ms=120,
            cost_usd=0.001,
        )
    ]
    return AgentRunResult(
        dataset=dataset,
        cases=cases,
        runner_id="tests:DummyAgent",
        run_id=run_id,
    )


class TestAgentRun:
    def test_run_succeeds(self, cli_env, agent_dataset_with_cases):
        fake_result = _make_fake_agent_run_result()
        with patch("supereval.agent.cli.load_runner") as mock_load:
            with patch("supereval.agent.cli.run_agent_eval", return_value=fake_result):
                mock_load.return_value = object()  # any truthy value
                result = invoke(
                    "run", "test-agent",
                    "--runner", "tests:DummyAgent",
                    "--no-record",
                    env=cli_env,
                )
        assert result.exit_code == 0
        assert "agent_test001" in result.output
        assert "100.0%" in result.output

    def test_run_requires_runner(self, cli_env, agent_meta):
        result = invoke("run", "test-agent", env=cli_env)
        assert result.exit_code != 0

    def test_run_fails_on_bad_runner(self, cli_env, agent_dataset_with_cases):
        with patch("supereval.agent.cli.load_runner",
                   side_effect=ImportError("no module")):
            result = invoke(
                "run", "test-agent",
                "--runner", "bad.module:Cls",
                env=cli_env,
            )
        assert result.exit_code != 0
        assert "Failed to load runner" in result.output

    def test_update_baseline(self, cli_env, agent_dataset_with_cases):
        fake_result = _make_fake_agent_run_result()
        with patch("supereval.agent.cli.load_runner") as mock_load:
            with patch("supereval.agent.cli.run_agent_eval", return_value=fake_result):
                mock_load.return_value = object()
                result = invoke(
                    "run", "test-agent",
                    "--runner", "tests:DummyAgent",
                    "--update-baseline",
                    "--no-record",
                    env=cli_env,
                )
        assert result.exit_code == 0
        assert "Baseline updated" in result.output
        from supereval.agent.baseline import load_agent_baseline
        baseline = load_agent_baseline("test-agent")
        assert baseline is not None
        assert baseline["run_id"] == "agent_test001"

    def test_compare_baseline_no_baseline(self, cli_env, agent_dataset_with_cases):
        fake_result = _make_fake_agent_run_result()
        with patch("supereval.agent.cli.load_runner") as mock_load:
            with patch("supereval.agent.cli.run_agent_eval", return_value=fake_result):
                mock_load.return_value = object()
                result = invoke(
                    "run", "test-agent",
                    "--runner", "tests:DummyAgent",
                    "--compare-baseline",
                    "--no-record",
                    env=cli_env,
                )
        assert result.exit_code == 0
        assert "No baseline" in result.output

    def test_output_writes_json(self, cli_env, agent_dataset_with_cases, tmp_path):
        fake_result = _make_fake_agent_run_result()
        out_file = tmp_path / "results.json"
        with patch("supereval.agent.cli.load_runner") as mock_load:
            with patch("supereval.agent.cli.run_agent_eval", return_value=fake_result):
                mock_load.return_value = object()
                result = invoke(
                    "run", "test-agent",
                    "--runner", "tests:DummyAgent",
                    "--output", str(out_file),
                    "--no-record",
                    env=cli_env,
                )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["run_id"] == "agent_test001"


# ---------------------------------------------------------------------------
# history commands
# ---------------------------------------------------------------------------

def _seed_agent_run(env, run_id="agent_test001", dataset="test-agent"):
    import os
    os.environ["SUPEREVAL_DB_PATH"] = env["SUPEREVAL_DB_PATH"]
    from supereval.store import record_agent_run
    record_agent_run(_make_fake_agent_run_result(dataset=dataset, run_id=run_id))


class TestAgentHistoryList:
    def test_empty_shows_message(self, cli_env):
        result = invoke("history", "list", env=cli_env)
        assert result.exit_code == 0
        assert "No agent runs" in result.output

    def test_shows_recorded_run(self, cli_env, agent_meta):
        _seed_agent_run(cli_env)
        result = invoke("history", "list", env=cli_env)
        assert result.exit_code == 0
        assert "agent_test001" in result.output

    def test_filters_by_dataset(self, cli_env, agent_meta):
        _seed_agent_run(cli_env, run_id="agent_a", dataset="test-agent")
        _seed_agent_run(cli_env, run_id="agent_b", dataset="other-agent")
        result = invoke("history", "list", "--dataset", "test-agent", env=cli_env)
        assert result.exit_code == 0
        assert "agent_a" in result.output
        assert "agent_b" not in result.output


class TestAgentHistoryShow:
    def test_not_found(self, cli_env):
        result = invoke("history", "show", "nonexistent", env=cli_env)
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_shows_run(self, cli_env, agent_meta):
        _seed_agent_run(cli_env)
        result = invoke("history", "show", "agent_test001", env=cli_env)
        assert result.exit_code == 0
        assert "test-agent" in result.output


class TestAgentHistoryStats:
    def test_no_runs_shows_message(self, cli_env):
        result = invoke("history", "stats", "no-such-dataset", env=cli_env)
        assert result.exit_code == 0
        assert "No agent runs" in result.output

    def test_shows_stats_after_run(self, cli_env, agent_meta):
        _seed_agent_run(cli_env)
        result = invoke("history", "stats", "test-agent", env=cli_env)
        assert result.exit_code == 0
        assert "Pass rate" in result.output
