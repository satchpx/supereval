"""End-to-end CLI smoke tests using Typer's test runner."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from supereval.cli import app
from supereval.runner import CaseResult, RunResult

runner = CliRunner()


def invoke(*args, env=None):
    """Invoke the CLI app with the given args."""
    return runner.invoke(app, list(args), env=env)


@pytest.fixture
def cli_env(datasets_dir):
    """Environment dict with SUPEREVAL_DATASETS_DIR set to the test datasets dir."""
    return {"SUPEREVAL_DATASETS_DIR": str(datasets_dir)}


class TestDatasetCreate:
    def test_creates_dataset(self, cli_env):
        result = invoke("dataset", "create", "my-ds", "--type", "qa", env=cli_env)
        assert result.exit_code == 0
        assert "Created" in result.output

    def test_requires_type(self, cli_env):
        result = invoke("dataset", "create", "my-ds", env=cli_env)
        assert result.exit_code != 0

    def test_duplicate_name_fails(self, cli_env):
        invoke("dataset", "create", "my-ds", "--type", "qa", env=cli_env)
        result = invoke("dataset", "create", "my-ds", "--type", "qa", env=cli_env)
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_classification_with_labels(self, cli_env):
        result = invoke(
            "dataset", "create", "my-cls",
            "--type", "classification",
            "--labels", "a,b,c",
            env=cli_env,
        )
        assert result.exit_code == 0


class TestDatasetList:
    def test_empty_shows_no_datasets_message(self, cli_env):
        result = invoke("dataset", "list", env=cli_env)
        assert result.exit_code == 0
        assert "No datasets" in result.output

    def test_shows_created_datasets(self, cli_env):
        invoke("dataset", "create", "ds-one", "--type", "qa", env=cli_env)
        invoke("dataset", "create", "ds-two", "--type", "classification", env=cli_env)
        result = invoke("dataset", "list", env=cli_env)
        assert result.exit_code == 0
        assert "ds-one" in result.output
        assert "ds-two" in result.output


class TestDatasetShow:
    def test_shows_dataset_info(self, cli_env):
        invoke("dataset", "create", "my-ds", "--type", "qa",
               "--description", "Test description", env=cli_env)
        result = invoke("dataset", "show", "my-ds", env=cli_env)
        assert result.exit_code == 0
        assert "my-ds" in result.output
        assert "qa" in result.output

    def test_missing_dataset_fails(self, cli_env):
        result = invoke("dataset", "show", "nonexistent", env=cli_env)
        assert result.exit_code != 0


class TestDatasetAddCases:
    def test_add_valid_cases(self, cli_env, tmp_path):
        invoke("dataset", "create", "my-ds", "--type", "qa", env=cli_env)
        cases_file = tmp_path / "cases.jsonl"
        cases_file.write_text(
            '{"input":{"query":"What is S3?"},"expected":{"ground_truth":"Object storage"}}\n'
            '{"input":{"query":"What is EC2?"},"expected":{"ground_truth":"Virtual machines"}}\n'
        )
        result = invoke("dataset", "add-cases", "my-ds", "--from", str(cases_file), env=cli_env)
        assert result.exit_code == 0
        assert "Added 2 case(s)" in result.output

    def test_invalid_cases_rejected(self, cli_env, tmp_path):
        invoke("dataset", "create", "my-ds", "--type", "qa", env=cli_env)
        cases_file = tmp_path / "bad.jsonl"
        cases_file.write_text('{"bad":"data"}\n')
        result = invoke("dataset", "add-cases", "my-ds", "--from", str(cases_file), env=cli_env)
        assert result.exit_code != 0
        assert "validation error" in result.output.lower()

    def test_missing_file_fails(self, cli_env):
        invoke("dataset", "create", "my-ds", "--type", "qa", env=cli_env)
        result = invoke("dataset", "add-cases", "my-ds", "--from", "/nonexistent.jsonl", env=cli_env)
        assert result.exit_code != 0


class TestDatasetValidate:
    def test_valid_dataset_passes(self, cli_env, qa_dataset_with_cases):
        result = invoke("dataset", "validate", "test-qa", env=cli_env)
        assert result.exit_code == 0
        assert "OK" in result.output
        assert "3" in result.output

    def test_empty_dataset_passes(self, cli_env, qa_meta):
        result = invoke("dataset", "validate", "test-qa", env=cli_env)
        assert result.exit_code == 0
        assert "0" in result.output


class TestDatasetExport:
    def test_exports_yaml_to_stdout(self, cli_env, qa_dataset_with_cases):
        result = invoke("dataset", "export", "test-qa", env=cli_env)
        assert result.exit_code == 0
        assert "tests:" in result.output
        assert "query:" in result.output

    def test_exports_to_file(self, cli_env, qa_dataset_with_cases, tmp_path):
        out_file = tmp_path / "output.yaml"
        result = invoke("dataset", "export", "test-qa", "--output", str(out_file), env=cli_env)
        assert result.exit_code == 0
        assert out_file.exists()
        assert "tests:" in out_file.read_text()


class TestRun:
    def _make_run_result(self, dataset: str) -> RunResult:
        return RunResult(
            dataset=dataset,
            providers=["anthropic:claude-opus-4-6"],
            cases=[
                CaseResult(vars={"query": "q1"}, passed=True, score=1.0),
                CaseResult(vars={"query": "q2"}, passed=True, score=1.0),
            ],
            run_id="run_test",
        )

    def test_run_requires_model_or_config(self, cli_env, qa_dataset_with_cases):
        result = invoke("run", "test-qa", env=cli_env)
        assert result.exit_code != 0
        assert "--model" in result.output or "config" in result.output.lower()

    def test_run_with_model(self, cli_env, qa_dataset_with_cases):
        mock_result = self._make_run_result("test-qa")
        with patch("supereval.cli.run_eval", return_value=mock_result):
            result = invoke("run", "test-qa", "--model", "anthropic:claude-opus-4-6", env=cli_env)
        assert result.exit_code == 0
        assert "2/2" in result.output

    def test_run_compare_baseline_no_baseline_warns(self, cli_env, qa_dataset_with_cases):
        mock_result = self._make_run_result("test-qa")
        with patch("supereval.cli.run_eval", return_value=mock_result):
            result = invoke(
                "run", "test-qa",
                "--model", "anthropic:claude-opus-4-6",
                "--compare-baseline",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "No baseline" in result.output

    def test_run_update_baseline(self, cli_env, qa_dataset_with_cases, datasets_dir):
        mock_result = self._make_run_result("test-qa")
        with patch("supereval.cli.run_eval", return_value=mock_result):
            result = invoke(
                "run", "test-qa",
                "--model", "anthropic:claude-opus-4-6",
                "--update-baseline",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "Baseline updated" in result.output
        assert (datasets_dir / "test-qa" / "baseline.json").exists()

    def test_run_saves_output_file(self, cli_env, qa_dataset_with_cases, tmp_path):
        mock_result = self._make_run_result("test-qa")
        out_file = tmp_path / "results.json"
        with patch("supereval.cli.run_eval", return_value=mock_result):
            result = invoke(
                "run", "test-qa",
                "--model", "anthropic:claude-opus-4-6",
                "--output", str(out_file),
                env=cli_env,
            )
        assert result.exit_code == 0
        assert out_file.exists()
        data = json.loads(out_file.read_text())
        assert data["run_id"] == "run_test"


class TestDatasetVersion:
    def test_tag_creates_version(self, cli_env, qa_meta):
        result = invoke("dataset", "version", "tag", "test-qa", "v1.0.0", env=cli_env)
        assert result.exit_code == 0
        assert "v1.0.0" in result.output

    def test_tag_with_description(self, cli_env, qa_meta):
        result = invoke(
            "dataset", "version", "tag", "test-qa", "v1.0.0",
            "--description", "first release",
            env=cli_env,
        )
        assert result.exit_code == 0
        assert "first release" in result.output

    def test_tag_duplicate_fails(self, cli_env, qa_meta):
        invoke("dataset", "version", "tag", "test-qa", "v1.0.0", env=cli_env)
        result = invoke("dataset", "version", "tag", "test-qa", "v1.0.0", env=cli_env)
        assert result.exit_code != 0
        assert "already exists" in result.output

    def test_tag_invalid_format_fails(self, cli_env, qa_meta):
        result = invoke("dataset", "version", "tag", "test-qa", "1.0.0", env=cli_env)
        assert result.exit_code != 0
        assert "Invalid version" in result.output

    def test_list_empty(self, cli_env, qa_meta):
        result = invoke("dataset", "version", "list", "test-qa", env=cli_env)
        assert result.exit_code == 0
        assert "No versions" in result.output

    def test_list_shows_versions(self, cli_env, qa_meta):
        invoke("dataset", "version", "tag", "test-qa", "v1.0.0", env=cli_env)
        invoke("dataset", "version", "tag", "test-qa", "v1.1.0", env=cli_env)
        result = invoke("dataset", "version", "list", "test-qa", env=cli_env)
        assert result.exit_code == 0
        assert "v1.0.0" in result.output
        assert "v1.1.0" in result.output

    def test_show_version(self, cli_env, qa_meta, qa_cases):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        invoke("dataset", "version", "tag", "test-qa", "v1.0.0", env=cli_env)
        result = invoke("dataset", "version", "show", "test-qa", "v1.0.0", env=cli_env)
        assert result.exit_code == 0
        assert "v1.0.0" in result.output

    def test_show_missing_version_fails(self, cli_env, qa_meta):
        result = invoke("dataset", "version", "show", "test-qa", "v9.9.9", env=cli_env)
        assert result.exit_code != 0

    def test_restore_with_yes_flag(self, cli_env, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases, load_cases
        append_cases("test-qa", qa_cases)
        invoke("dataset", "version", "tag", "test-qa", "v1.0.0", env=cli_env)
        # Add an extra case after tagging
        append_cases("test-qa", [qa_cases[0]])
        assert len(load_cases("test-qa")) == len(qa_cases) + 1
        result = invoke(
            "dataset", "version", "restore", "test-qa", "v1.0.0",
            "--yes",
            env=cli_env,
        )
        assert result.exit_code == 0
        assert "Restored" in result.output
        assert len(load_cases("test-qa")) == len(qa_cases)

    def test_restore_missing_version_fails(self, cli_env, qa_meta):
        result = invoke(
            "dataset", "version", "restore", "test-qa", "v9.9.9",
            "--yes",
            env=cli_env,
        )
        assert result.exit_code != 0


class TestGenerate:
    def test_fails_for_missing_path(self, cli_env, qa_meta):
        result = invoke(
            "generate", "test-qa",
            "--from", "/nonexistent/path",
            env=cli_env,
        )
        assert result.exit_code != 0

    def test_generates_staged_file(self, cli_env, qa_meta, doc_dir, tmp_path):
        from supereval.generator import GeneratedCase

        fake_cases = [
            GeneratedCase(
                description="S3 size limit",
                input={"query": "What is the max S3 object size?"},
                expected={"ground_truth": "5 TB"},
                tags=["s3"],
                difficulty="easy",
                case_type="factual",
                source_excerpt="Maximum object size is 5 TB.",
            )
        ]
        staged = tmp_path / "staged.jsonl"
        with patch("supereval.cli.BedrockGenerator") as MockGen:
            MockGen.return_value.generate.return_value = fake_cases
            result = invoke(
                "generate", "test-qa",
                "--from", str(doc_dir / "s3-guide.md"),
                "--count", "1",
                "--output", str(staged),
                env=cli_env,
            )
        assert result.exit_code == 0
        assert staged.exists()
        line = json.loads(staged.read_text().strip())
        assert line["input"]["query"] == "What is the max S3 object size?"
        assert "source_excerpt" in line
        assert "case_type" in line

    def test_auto_import_adds_cases_to_dataset(self, cli_env, qa_meta, doc_dir):
        from supereval.generator import GeneratedCase

        fake_cases = [
            GeneratedCase(
                description="Lambda timeout",
                input={"query": "What is the max Lambda timeout?"},
                expected={"ground_truth": "15 minutes"},
                tags=["lambda"],
                difficulty="easy",
                case_type="factual",
                source_excerpt="Maximum timeout is 15 minutes.",
            )
        ]
        with patch("supereval.cli.BedrockGenerator") as MockGen:
            MockGen.return_value.generate.return_value = fake_cases
            result = invoke(
                "generate", "test-qa",
                "--from", str(doc_dir / "lambda-guide.txt"),
                "--count", "1",
                "--auto-import",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "Imported 1 case" in result.output

    def test_openai_backend_uses_openai_generator(self, cli_env, qa_meta, doc_dir, tmp_path):
        from supereval.generator import GeneratedCase

        fake_cases = [
            GeneratedCase(
                description="S3 size",
                input={"query": "Max S3 size?"},
                expected={"ground_truth": "5 TB"},
                tags=[], difficulty="easy", case_type="factual", source_excerpt="",
            )
        ]
        staged = tmp_path / "staged.jsonl"
        with patch("supereval.cli.OpenAIGenerator") as MockGen:
            MockGen.return_value.generate.return_value = fake_cases
            result = invoke(
                "generate", "test-qa",
                "--from", str(doc_dir / "s3-guide.md"),
                "--backend", "openai",
                "--output", str(staged),
                env=cli_env,
            )
        assert result.exit_code == 0
        MockGen.assert_called_once()

    def test_unknown_backend_fails(self, cli_env, qa_meta, doc_dir):
        result = invoke(
            "generate", "test-qa",
            "--from", str(doc_dir),
            "--backend", "unknown-backend",
            env=cli_env,
        )
        assert result.exit_code != 0
        assert "Unknown backend" in result.output

    def test_interactive_keep_all(self, cli_env, qa_meta, doc_dir, tmp_path):
        from supereval.generator import GeneratedCase

        fake_cases = [
            GeneratedCase(
                description="S3 size limit",
                input={"query": "What is the max S3 object size?"},
                expected={"ground_truth": "5 TB"},
                tags=["s3"], difficulty="easy", case_type="factual",
                source_excerpt="Maximum object size is 5 TB.",
            ),
            GeneratedCase(
                description="Lambda timeout",
                input={"query": "What is the max Lambda timeout?"},
                expected={"ground_truth": "15 minutes"},
                tags=["lambda"], difficulty="easy", case_type="factual",
                source_excerpt="Max timeout is 15 minutes.",
            ),
        ]
        staged = tmp_path / "staged.jsonl"
        with patch("supereval.cli.BedrockGenerator") as MockGen:
            MockGen.return_value.generate.return_value = fake_cases
            # "k\nk\n" — keep both cases
            result = runner.invoke(
                app,
                ["generate", "test-qa",
                 "--from", str(doc_dir / "s3-guide.md"),
                 "--count", "2",
                 "--output", str(staged),
                 "--interactive"],
                input="k\nk\n",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "Review complete" in result.output
        assert "2 case(s) approved" in result.output
        lines = [l for l in staged.read_text().strip().splitlines() if l]
        assert len(lines) == 2

    def test_interactive_skip_all_exits_cleanly(self, cli_env, qa_meta, doc_dir, tmp_path):
        from supereval.generator import GeneratedCase

        fake_cases = [
            GeneratedCase(
                description="S3 size limit",
                input={"query": "Max S3 size?"},
                expected={"ground_truth": "5 TB"},
                tags=[], difficulty="easy", case_type="factual",
                source_excerpt="",
            ),
        ]
        staged = tmp_path / "staged.jsonl"
        with patch("supereval.cli.BedrockGenerator") as MockGen:
            MockGen.return_value.generate.return_value = fake_cases
            # "s\n" — skip the only case
            result = runner.invoke(
                app,
                ["generate", "test-qa",
                 "--from", str(doc_dir / "s3-guide.md"),
                 "--count", "1",
                 "--output", str(staged),
                 "--interactive"],
                input="s\n",
                env=cli_env,
            )
        assert result.exit_code == 0
        assert "No cases approved" in result.output
        assert not staged.exists()

    def test_interactive_quit_stops_early(self, cli_env, qa_meta, doc_dir, tmp_path):
        from supereval.generator import GeneratedCase

        fake_cases = [
            GeneratedCase(
                description="Case 1",
                input={"query": "Q1?"}, expected={"ground_truth": "A1"},
                tags=[], difficulty="easy", case_type="factual", source_excerpt="",
            ),
            GeneratedCase(
                description="Case 2",
                input={"query": "Q2?"}, expected={"ground_truth": "A2"},
                tags=[], difficulty="easy", case_type="factual", source_excerpt="",
            ),
        ]
        staged = tmp_path / "staged.jsonl"
        with patch("supereval.cli.BedrockGenerator") as MockGen:
            MockGen.return_value.generate.return_value = fake_cases
            # keep case 1, then quit before case 2
            result = runner.invoke(
                app,
                ["generate", "test-qa",
                 "--from", str(doc_dir / "s3-guide.md"),
                 "--count", "2",
                 "--output", str(staged),
                 "--interactive"],
                input="k\nq\n",
                env=cli_env,
            )
        assert result.exit_code == 0
        lines = [l for l in staged.read_text().strip().splitlines() if l]
        assert len(lines) == 1
        assert json.loads(lines[0])["description"] == "Case 1"


# ---------------------------------------------------------------------------
# run-all subcommand tests
# ---------------------------------------------------------------------------

def _make_pass_result(dataset: str) -> "RunResult":
    from supereval.runner import CaseResult, RunResult
    return RunResult(
        dataset=dataset,
        providers=["test-model"],
        cases=[CaseResult(vars={"query": "q1"}, passed=True, score=1.0)],
        run_id=f"run_{dataset}",
    )


def _make_fail_result(dataset: str) -> "RunResult":
    from supereval.runner import CaseResult, RunResult
    return RunResult(
        dataset=dataset,
        providers=["test-model"],
        cases=[CaseResult(vars={"query": "q1"}, passed=False, score=0.0)],
        run_id=f"run_{dataset}",
    )


class TestRunAll:
    def test_requires_model_or_config(self, cli_env):
        result = invoke("run-all", env=cli_env)
        assert result.exit_code != 0

    def test_no_datasets_exits_cleanly(self, cli_env):
        result = invoke("run-all", "--model", "test-model", env=cli_env)
        assert result.exit_code == 0
        assert "No datasets" in result.output

    def test_all_pass_exits_zero(self, cli_env, qa_dataset_with_cases):
        with patch("supereval.cli.run_eval", return_value=_make_pass_result("test-qa")):
            result = invoke("run-all", "--model", "test-model", env=cli_env)
        assert result.exit_code == 0
        assert "All datasets passed" in result.output

    def test_any_failure_exits_nonzero(self, cli_env, qa_dataset_with_cases):
        with patch("supereval.cli.run_eval", return_value=_make_fail_result("test-qa")):
            result = invoke("run-all", "--model", "test-model", env=cli_env)
        assert result.exit_code != 0
        assert "failed" in result.output.lower()

    def test_summary_table_lists_all_datasets(self, cli_env, qa_dataset_with_cases):
        with patch("supereval.cli.run_eval", return_value=_make_pass_result("test-qa")):
            result = invoke("run-all", "--model", "test-model", env=cli_env)
        assert "test-qa" in result.output

    def test_update_baseline_after_run(self, cli_env, qa_dataset_with_cases, datasets_dir):
        with patch("supereval.cli.run_eval", return_value=_make_pass_result("test-qa")):
            result = invoke(
                "run-all", "--model", "test-model", "--update-baseline", env=cli_env
            )
        assert result.exit_code == 0
        assert (datasets_dir / "test-qa" / "baseline.json").exists()

    def test_output_dir_writes_json_files(self, cli_env, qa_dataset_with_cases, tmp_path):
        out_dir = tmp_path / "results"
        with patch("supereval.cli.run_eval", return_value=_make_pass_result("test-qa")):
            result = invoke(
                "run-all", "--model", "test-model",
                "--output-dir", str(out_dir),
                env=cli_env,
            )
        assert result.exit_code == 0
        assert (out_dir / "test-qa.json").exists()
        data = json.loads((out_dir / "test-qa.json").read_text())
        assert data["dataset"] == "test-qa"

    def test_run_error_continues_and_exits_nonzero(self, cli_env, qa_dataset_with_cases):
        with patch("supereval.cli.run_eval", side_effect=RuntimeError("promptfoo failed")):
            result = invoke("run-all", "--model", "test-model", env=cli_env)
        assert result.exit_code != 0
        assert "ERROR" in result.output


# ---------------------------------------------------------------------------
# history subcommand tests
# ---------------------------------------------------------------------------

@pytest.fixture
def cli_env_with_db(tmp_path, datasets_dir):
    """CLI environment with isolated datasets dir AND isolated DuckDB."""
    return {
        "SUPEREVAL_DATASETS_DIR": str(datasets_dir),
        "SUPEREVAL_DB_PATH": str(tmp_path / "test.db"),
    }


def _fake_run_result(run_id: str = "run_test001", dataset: str = "test-qa"):
    from supereval.runner import CaseResult, RunResult
    return RunResult(
        dataset=dataset,
        providers=["test-model"],
        cases=[
            CaseResult(vars={"query": "q1"}, passed=True, score=1.0, latency_ms=100, cost_usd=0.001),
            CaseResult(vars={"query": "q2"}, passed=False, score=0.0, latency_ms=200, cost_usd=0.002),
        ],
        run_id=run_id,
        total_tokens=500,
        prompt_tokens=300,
        completion_tokens=200,
    )


def _seed_db(env: dict, run_id: str = "run_test001", dataset: str = "test-qa"):
    """Directly record a run into the isolated DB."""
    import os
    os.environ["SUPEREVAL_DB_PATH"] = env["SUPEREVAL_DB_PATH"]
    from supereval.store import record_run
    record_run(_fake_run_result(run_id=run_id, dataset=dataset))


class TestHistoryList:
    def test_empty_when_no_runs(self, cli_env_with_db):
        result = invoke("history", "list", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "No runs" in result.output

    def test_shows_recorded_run(self, cli_env_with_db):
        _seed_db(cli_env_with_db)
        result = invoke("history", "list", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "run_test001" in result.output

    def test_filters_by_dataset(self, cli_env_with_db):
        _seed_db(cli_env_with_db, run_id="run_qa", dataset="qa-ds")
        _seed_db(cli_env_with_db, run_id="run_cls", dataset="cls-ds")
        result = invoke("history", "list", "--dataset", "qa-ds", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "run_qa" in result.output
        assert "run_cls" not in result.output


class TestHistoryShow:
    def test_run_not_found(self, cli_env_with_db):
        result = invoke("history", "show", "nonexistent", env=cli_env_with_db)
        assert result.exit_code != 0
        assert "not found" in result.output

    def test_shows_run_summary(self, cli_env_with_db):
        _seed_db(cli_env_with_db)
        result = invoke("history", "show", "run_test001", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "run_test001" in result.output
        assert "test-qa" in result.output

    def test_shows_cases_with_flag(self, cli_env_with_db):
        _seed_db(cli_env_with_db)
        result = invoke("history", "show", "run_test001", "--cases", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "q1" in result.output or "Case results" in result.output


class TestHistoryStats:
    def test_no_runs_shows_message(self, cli_env_with_db):
        result = invoke("history", "stats", "no-such-dataset", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "No runs" in result.output

    def test_shows_stats_after_run(self, cli_env_with_db):
        _seed_db(cli_env_with_db)
        result = invoke("history", "stats", "test-qa", env=cli_env_with_db)
        assert result.exit_code == 0
        assert "Pass rate" in result.output


class TestRunNoRecord:
    def _mock_promptfoo(self, fake_output: dict):
        def fake_run(cmd, capture_output, text):
            out_idx = cmd.index("--output") + 1
            Path(cmd[out_idx]).write_text(json.dumps(fake_output))
            return MagicMock(returncode=0, stdout="", stderr="")
        return fake_run

    def _fake_promptfoo_output(self):
        return {"results": {"results": [{"vars": {"query": "q1"}, "success": True, "score": 1.0, "latencyMs": 100}]}}

    def test_no_record_skips_db(self, cli_env_with_db, qa_dataset_with_cases):
        with patch("supereval.runner.subprocess.run", side_effect=self._mock_promptfoo(self._fake_promptfoo_output())):
            with patch("supereval.runner.shutil.which", return_value="/usr/local/bin/promptfoo"):
                result = invoke(
                    "run", "test-qa",
                    "--model", "test-model",
                    "--no-record",
                    env=cli_env_with_db,
                )
        assert result.exit_code == 0
        # DB should be empty
        import os
        os.environ["SUPEREVAL_DB_PATH"] = cli_env_with_db["SUPEREVAL_DB_PATH"]
        from supereval.store import list_runs
        assert list_runs() == []

    def test_record_persists_by_default(self, cli_env_with_db, qa_dataset_with_cases):
        with patch("supereval.runner.subprocess.run", side_effect=self._mock_promptfoo(self._fake_promptfoo_output())):
            with patch("supereval.runner.shutil.which", return_value="/usr/local/bin/promptfoo"):
                result = invoke(
                    "run", "test-qa",
                    "--model", "test-model",
                    env=cli_env_with_db,
                )
        assert result.exit_code == 0
        import os
        os.environ["SUPEREVAL_DB_PATH"] = cli_env_with_db["SUPEREVAL_DB_PATH"]
        from supereval.store import list_runs
        runs = list_runs()
        assert len(runs) == 1
