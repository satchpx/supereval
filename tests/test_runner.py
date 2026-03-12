"""Tests for eval runner: config building, output parsing, tier resolution."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from supereval.runner import (
    _build_promptfoo_config,
    _parse_promptfoo_output,
    _vars_key,
    run_eval,
)


def _make_promptfoo_output(results: list[dict]) -> dict:
    """Build a minimal Promptfoo-shaped output dict."""
    return {"results": {"results": results}}


def _make_result_item(vars: dict, success: bool, score: float = None, latency: int = 100, cost: float = 0.0) -> dict:
    return {
        "vars": vars,
        "success": success,
        "score": score if score is not None else (1.0 if success else 0.0),
        "latencyMs": latency,
        "cost": cost,
    }


class TestVarsKey:
    def test_same_vars_same_key(self):
        v = {"query": "What is S3?"}
        assert _vars_key(v) == _vars_key(v)

    def test_different_vars_different_key(self):
        assert _vars_key({"query": "a"}) != _vars_key({"query": "b"})

    def test_key_order_independent(self):
        v1 = {"a": 1, "b": 2}
        v2 = {"b": 2, "a": 1}
        assert _vars_key(v1) == _vars_key(v2)


class TestBuildPromptfooConfig:
    def test_tier1_auto_generates_prompt_and_providers(self, qa_dataset_with_cases):
        config = _build_promptfoo_config(
            dataset_name="test-qa",
            models=["anthropic:claude-opus-4-6"],
            prompt=None,
            customer_config=None,
        )
        assert "prompts" in config
        assert "providers" in config
        assert config["providers"] == ["anthropic:claude-opus-4-6"]
        assert "tests" in config
        assert len(config["tests"]) == 3
        # Default prompt should reference {{query}}
        assert "{{query}}" in config["prompts"][0]

    def test_tier2_uses_custom_prompt(self, qa_dataset_with_cases):
        custom_prompt = "You are an AWS expert. Answer: {{query}}"
        config = _build_promptfoo_config(
            dataset_name="test-qa",
            models=["anthropic:claude-opus-4-6"],
            prompt=custom_prompt,
            customer_config=None,
        )
        assert config["prompts"] == [custom_prompt]

    def test_tier3_injects_tests_into_customer_config(self, qa_dataset_with_cases):
        customer_config = {
            "prompts": ["Custom: {{query}}"],
            "providers": ["openai:gpt-4o"],
            "defaultTest": {"options": {"timeout": 30000}},
        }
        config = _build_promptfoo_config(
            dataset_name="test-qa",
            models=[],
            prompt=None,
            customer_config=customer_config,
        )
        assert config["prompts"] == ["Custom: {{query}}"]
        assert config["providers"] == ["openai:gpt-4o"]
        assert "defaultTest" in config
        assert len(config["tests"]) == 3

    def test_tier3_does_not_mutate_customer_config(self, qa_dataset_with_cases):
        original = {"prompts": ["p"], "providers": ["m"]}
        import copy
        customer_copy = copy.deepcopy(original)
        _build_promptfoo_config("test-qa", [], None, customer_copy)
        assert "tests" not in original

    def test_classification_default_prompt_contains_labels(self, classification_meta):
        from supereval.models import ClassificationTestCase, ClassificationInput, ClassificationExpected
        from supereval.storage import append_cases
        append_cases("test-classification", [
            ClassificationTestCase(
                input=ClassificationInput(text="my instance won't start"),
                expected=ClassificationExpected(label="compute"),
            )
        ])
        config = _build_promptfoo_config(
            dataset_name="test-classification",
            models=["anthropic:claude-opus-4-6"],
            prompt=None,
            customer_config=None,
        )
        # Labels from dataset should be baked into the prompt
        assert "networking" in config["prompts"][0]
        assert "compute" in config["prompts"][0]
        assert "storage" in config["prompts"][0]


class TestParsePromptfooOutput:
    def test_parses_single_provider(self, qa_meta):
        raw = _make_promptfoo_output([
            _make_result_item({"query": "What is S3?"}, success=True),
            _make_result_item({"query": "What is Lambda?"}, success=False),
        ])
        result = _parse_promptfoo_output(raw, ["model-a"], "test-qa")
        assert result.total == 2
        assert result.passed == 1
        assert result.failed == 1

    def test_multi_provider_and_semantics(self, qa_meta):
        """A case passes only if ALL providers pass."""
        vars_ = {"query": "What is S3?"}
        raw = _make_promptfoo_output([
            _make_result_item(vars_, success=True),   # provider A passes
            _make_result_item(vars_, success=False),  # provider B fails
        ])
        result = _parse_promptfoo_output(raw, ["model-a", "model-b"], "test-qa")
        assert result.total == 1  # same vars → same case
        assert result.passed == 0  # AND semantics: one failure → case fails

    def test_score_is_minimum_across_providers(self, qa_meta):
        vars_ = {"query": "test?"}
        raw = _make_promptfoo_output([
            _make_result_item(vars_, success=True, score=0.9),
            _make_result_item(vars_, success=True, score=0.6),
        ])
        result = _parse_promptfoo_output(raw, ["a", "b"], "test-qa")
        assert result.cases[0].score == pytest.approx(0.6)

    def test_latency_is_averaged_across_providers(self, qa_meta):
        vars_ = {"query": "test?"}
        raw = _make_promptfoo_output([
            _make_result_item(vars_, success=True, latency=200),
            _make_result_item(vars_, success=True, latency=400),
        ])
        result = _parse_promptfoo_output(raw, ["a", "b"], "test-qa")
        assert result.cases[0].latency_ms == 300

    def test_providers_stored_on_result(self, qa_meta):
        raw = _make_promptfoo_output([_make_result_item({"query": "q"}, True)])
        providers = ["anthropic:claude-opus-4-6", "openai:gpt-4o"]
        result = _parse_promptfoo_output(raw, providers, "test-qa")
        assert result.providers == providers

    def test_cost_extracted_per_case(self, qa_meta):
        raw = _make_promptfoo_output([
            _make_result_item({"query": "q1"}, success=True, cost=0.001),
            _make_result_item({"query": "q2"}, success=True, cost=0.002),
        ])
        result = _parse_promptfoo_output(raw, ["model-a"], "test-qa")
        total = sum(c.cost_usd for c in result.cases)
        assert total == pytest.approx(0.003, abs=1e-6)

    def test_cost_summed_across_providers(self, qa_meta):
        """When multiple providers run the same case, costs are summed."""
        vars_ = {"query": "q"}
        raw = _make_promptfoo_output([
            _make_result_item(vars_, success=True, cost=0.001),
            _make_result_item(vars_, success=True, cost=0.002),
        ])
        result = _parse_promptfoo_output(raw, ["a", "b"], "test-qa")
        assert result.cases[0].cost_usd == pytest.approx(0.003, abs=1e-6)

    def test_token_usage_extracted_from_stats(self, qa_meta):
        raw = {
            "results": {
                "results": [_make_result_item({"query": "q"}, True)],
                "stats": {
                    "tokenUsage": {"total": 500, "prompt": 300, "completion": 200}
                },
            }
        }
        result = _parse_promptfoo_output(raw, ["model-a"], "test-qa")
        assert result.total_tokens == 500
        assert result.prompt_tokens == 300
        assert result.completion_tokens == 200

    def test_total_cost_usd_property(self, qa_meta):
        raw = _make_promptfoo_output([
            _make_result_item({"query": "q1"}, True, cost=0.005),
            _make_result_item({"query": "q2"}, False, cost=0.003),
        ])
        result = _parse_promptfoo_output(raw, ["model-a"], "test-qa")
        assert result.total_cost_usd == pytest.approx(0.008, abs=1e-6)

    def test_latency_percentile_properties(self, qa_meta):
        raw = _make_promptfoo_output([
            _make_result_item({"query": f"q{i}"}, True, latency=lat)
            for i, lat in enumerate([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
        ])
        result = _parse_promptfoo_output(raw, ["model-a"], "test-qa")
        assert result.avg_latency_ms == pytest.approx(550.0, abs=1.0)
        assert result.p95_latency_ms >= 900

    def test_run_result_to_dict(self, qa_meta, sample_run_result):
        d = sample_run_result.to_dict()
        assert d["dataset"] == "test-qa"
        assert d["summary"]["total"] == 3
        assert d["summary"]["passed"] == 2
        assert d["summary"]["pass_rate"] == pytest.approx(2 / 3, abs=0.01)
        assert len(d["cases"]) == 3


class TestRunEval:
    def _mock_promptfoo(self, tmp_results: dict):
        """Returns a mock for subprocess.run that writes results JSON to the output path."""
        def fake_run(cmd, capture_output, text):
            # Find --output arg and write fake results there
            out_idx = cmd.index("--output") + 1
            Path(cmd[out_idx]).write_text(json.dumps(tmp_results))
            return MagicMock(returncode=0, stdout="", stderr="")
        return fake_run

    def test_tier1_runs_successfully(self, qa_dataset_with_cases):
        fake_output = _make_promptfoo_output([
            _make_result_item({"query": "What is the max SQS message size?"}, True),
        ])
        with patch("supereval.runner.subprocess.run", side_effect=self._mock_promptfoo(fake_output)):
            with patch("supereval.runner.shutil.which", return_value="/usr/local/bin/promptfoo"):
                result = run_eval("test-qa", models=["anthropic:claude-opus-4-6"])
        assert result.total >= 1

    def test_raises_without_models_or_config(self, qa_dataset_with_cases):
        with pytest.raises(ValueError, match="No models"):
            run_eval("test-qa")

    def test_raises_when_promptfoo_not_found(self, qa_dataset_with_cases):
        with patch("supereval.runner.shutil.which", return_value=None):
            with pytest.raises(RuntimeError, match="promptfoo not found"):
                run_eval("test-qa", models=["some-model"])

    def test_raises_when_results_file_missing(self, qa_dataset_with_cases):
        def fake_run_no_output(cmd, capture_output, text):
            return MagicMock(returncode=1, stdout="", stderr="error")

        with patch("supereval.runner.subprocess.run", side_effect=fake_run_no_output):
            with patch("supereval.runner.shutil.which", return_value="/usr/local/bin/promptfoo"):
                with pytest.raises(RuntimeError, match="failed"):
                    run_eval("test-qa", models=["some-model"])
