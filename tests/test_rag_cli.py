"""End-to-end CLI smoke tests for supereval rag commands."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from supereval.cli import app
from supereval.rag.models import RagCaseResult, RagRunResult, RagScore
from supereval.rag.storage import save_rag_dataset_meta, append_rag_cases
from supereval.rag.models import RagDatasetMeta, RagTestCase, RagInput, RagExpected

runner = CliRunner()


def invoke(*args, env=None):
    return runner.invoke(app, list(args), env=env)


@pytest.fixture
def cli_env(datasets_dir):
    return {"SUPEREVAL_DATASETS_DIR": str(datasets_dir)}


@pytest.fixture
def rag_meta(datasets_dir, cli_env):
    meta = RagDatasetMeta(name="test-rag")
    save_rag_dataset_meta(meta)
    return meta


@pytest.fixture
def rag_dataset_with_cases(rag_meta):
    cases = [
        RagTestCase(
            id="rtc_001",
            input=RagInput(
                query="What region is my-bucket in?",
                retrieved_contexts=["my-bucket is in us-west-2."],
            ),
            expected=RagExpected(ground_truth="us-west-2"),
        ),
    ]
    append_rag_cases("test-rag", cases)
    return rag_meta


def _make_run_result(dataset: str = "test-rag") -> RagRunResult:
    score = RagScore(
        contains_score=1.0,
        answer_correctness_score=1.0,
        faithfulness_score=1.0,
        composite_score=1.0,
        passed=True,
    )
    return RagRunResult(
        dataset=dataset,
        cases=[
            RagCaseResult(
                case_id="rtc_001",
                description="Region lookup",
                vars={"query": "What region?"},
                answer="us-west-2",
                score=score,
                latency_ms=100,
            )
        ],
        model_id="test-model",
        run_id="rag_test001",
    )


class TestRagDatasetCreate:
    def test_creates_dataset(self, cli_env):
        result = invoke("rag", "dataset", "create", "my-rag", env=cli_env)
        assert result.exit_code == 0
        assert "Created" in result.output

    def test_duplicate_name_fails(self, cli_env):
        invoke("rag", "dataset", "create", "my-rag", env=cli_env)
        result = invoke("rag", "dataset", "create", "my-rag", env=cli_env)
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_with_tags_and_description(self, cli_env):
        result = invoke(
            "rag", "dataset", "create", "my-rag",
            "--description", "Test RAG dataset",
            "--tags", "aws,s3",
            env=cli_env,
        )
        assert result.exit_code == 0


class TestRagDatasetList:
    def test_empty_shows_message(self, cli_env):
        result = invoke("rag", "dataset", "list", env=cli_env)
        assert result.exit_code == 0
        assert "No RAG datasets" in result.output

    def test_shows_created_datasets(self, cli_env):
        invoke("rag", "dataset", "create", "rag-one", env=cli_env)
        invoke("rag", "dataset", "create", "rag-two", env=cli_env)
        result = invoke("rag", "dataset", "list", env=cli_env)
        assert result.exit_code == 0
        assert "rag-one" in result.output
        assert "rag-two" in result.output


class TestRagDatasetShow:
    def test_shows_dataset_info(self, cli_env):
        invoke("rag", "dataset", "create", "my-rag", "--description", "A test", env=cli_env)
        result = invoke("rag", "dataset", "show", "my-rag", env=cli_env)
        assert result.exit_code == 0
        assert "my-rag" in result.output

    def test_missing_dataset_fails(self, cli_env):
        result = invoke("rag", "dataset", "show", "nonexistent", env=cli_env)
        assert result.exit_code != 0


class TestRagDatasetAddCases:
    def test_adds_cases_from_jsonl(self, cli_env, rag_meta, tmp_path):
        case = RagTestCase(
            input=RagInput(query="q?", retrieved_contexts=["ctx"]),
            expected=RagExpected(ground_truth="a"),
        )
        f = tmp_path / "cases.jsonl"
        f.write_text(case.model_dump_json() + "\n")
        result = invoke("rag", "dataset", "add-cases", "test-rag", "--from", str(f), env=cli_env)
        assert result.exit_code == 0
        assert "Added 1" in result.output

    def test_invalid_cases_rejected(self, cli_env, rag_meta, tmp_path):
        f = tmp_path / "bad.jsonl"
        f.write_text('{"invalid": true}\n')
        result = invoke("rag", "dataset", "add-cases", "test-rag", "--from", str(f), env=cli_env)
        assert result.exit_code != 0

    def test_missing_file_fails(self, cli_env, rag_meta):
        result = invoke(
            "rag", "dataset", "add-cases", "test-rag",
            "--from", "/nonexistent/cases.jsonl",
            env=cli_env,
        )
        assert result.exit_code != 0


class TestRagDatasetValidate:
    def test_valid_dataset_passes(self, cli_env, rag_dataset_with_cases):
        result = invoke("rag", "dataset", "validate", "test-rag", env=cli_env)
        assert result.exit_code == 0
        assert "OK" in result.output

    def test_empty_dataset_passes(self, cli_env, rag_meta):
        result = invoke("rag", "dataset", "validate", "test-rag", env=cli_env)
        assert result.exit_code == 0


class TestRagRun:
    def test_run_success(self, cli_env, rag_dataset_with_cases):
        run_result = _make_run_result()
        with patch("supereval.rag.cli.run_rag_eval", return_value=run_result):
            result = invoke(
                "rag", "run", "test-rag", "--model", "test-model", "--no-record",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "rag_test001" in result.output
        assert "1/1" in result.output

    def test_run_with_failing_cases_exits_zero_without_baseline(self, cli_env, rag_dataset_with_cases):
        score = RagScore(composite_score=0.0, passed=False, failure_reasons=["ground truth missing"])
        run_result = RagRunResult(
            dataset="test-rag",
            cases=[RagCaseResult(
                case_id="rtc_001", description="", vars={},
                answer="wrong", score=score,
            )],
            model_id="test-model",
        )
        with patch("supereval.rag.cli.run_rag_eval", return_value=run_result):
            result = invoke(
                "rag", "run", "test-rag", "--model", "test-model", "--no-record",
                env=cli_env,
            )
        assert result.exit_code == 0  # no --compare-baseline, just display

    def test_run_missing_dataset_fails(self, cli_env):
        result = invoke("rag", "run", "nonexistent", "--model", "test-model", env=cli_env)
        assert result.exit_code != 0

    def test_update_baseline(self, cli_env, rag_dataset_with_cases):
        run_result = _make_run_result()
        with patch("supereval.rag.cli.run_rag_eval", return_value=run_result):
            result = invoke(
                "rag", "run", "test-rag", "--model", "test-model",
                "--update-baseline", "--no-record",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "Baseline updated" in result.output

    def test_output_flag_writes_file(self, cli_env, rag_dataset_with_cases, tmp_path):
        run_result = _make_run_result()
        out = tmp_path / "results.json"
        with patch("supereval.rag.cli.run_rag_eval", return_value=run_result):
            result = invoke(
                "rag", "run", "test-rag", "--model", "test-model",
                "--output", str(out), "--no-record",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["run_id"] == "rag_test001"

    def test_compare_baseline_no_baseline_warns(self, cli_env, rag_dataset_with_cases):
        run_result = _make_run_result()
        with patch("supereval.rag.cli.run_rag_eval", return_value=run_result):
            result = invoke(
                "rag", "run", "test-rag", "--model", "test-model",
                "--compare-baseline", "--no-record",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "No baseline" in result.output

    def test_compare_baseline_regression_exits_nonzero(self, cli_env, rag_dataset_with_cases, tmp_path):
        # First run: set baseline with passing result
        passing = _make_run_result()
        with patch("supereval.rag.cli.run_rag_eval", return_value=passing):
            invoke(
                "rag", "run", "test-rag", "--model", "test-model",
                "--update-baseline", "--no-record",
                env=cli_env,
            )
        # Second run: failing result
        score = RagScore(composite_score=0.0, passed=False, failure_reasons=["missed"])
        failing = RagRunResult(
            dataset="test-rag",
            cases=[RagCaseResult(
                case_id="rtc_001", description="", vars={},
                answer="wrong", score=score,
            )],
            model_id="test-model",
        )
        with patch("supereval.rag.cli.run_rag_eval", return_value=failing):
            result = invoke(
                "rag", "run", "test-rag", "--model", "test-model",
                "--compare-baseline", "--no-record",
                env=cli_env,
            )
        assert result.exit_code == 1


class TestRagHistory:
    def test_history_list_empty(self, cli_env, monkeypatch):
        monkeypatch.setenv("SUPEREVAL_DB_PATH", str(Path(cli_env["SUPEREVAL_DATASETS_DIR"]) / "test.db"))
        result = invoke("rag", "history", "list", env=cli_env)
        assert result.exit_code == 0
        assert "No RAG runs" in result.output

    def test_history_stats_no_runs(self, cli_env, monkeypatch):
        monkeypatch.setenv("SUPEREVAL_DB_PATH", str(Path(cli_env["SUPEREVAL_DATASETS_DIR"]) / "test.db"))
        result = invoke("rag", "history", "stats", "test-rag", env=cli_env)
        assert result.exit_code == 0
        assert "No RAG runs" in result.output
