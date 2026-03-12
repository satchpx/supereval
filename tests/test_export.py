"""Tests for dataset → Promptfoo YAML export."""
from __future__ import annotations

import yaml

from supereval.export import _build_tests, export_to_promptfoo


class TestBuildTests:
    def test_qa_structure(self, qa_dataset_with_cases):
        tests = _build_tests("test-qa")
        assert len(tests) == 3
        t = tests[0]
        assert "vars" in t
        assert "query" in t["vars"]
        assert "assert" in t
        # QA cases get two assertions: icontains + llm-rubric
        types = [a["type"] for a in t["assert"]]
        assert "icontains" in types
        assert "llm-rubric" in types

    def test_qa_vars_match_cases(self, qa_dataset_with_cases, qa_cases):
        tests = _build_tests("test-qa")
        queries = [t["vars"]["query"] for t in tests]
        expected = [c.input.query for c in qa_cases]
        assert queries == expected

    def test_classification_structure(self, classification_meta):
        from supereval.models import ClassificationTestCase, ClassificationInput, ClassificationExpected
        from supereval.storage import append_cases
        cases = [
            ClassificationTestCase(
                input=ClassificationInput(text="EC2 won't start"),
                expected=ClassificationExpected(label="compute"),
            )
        ]
        append_cases("test-classification", cases)
        tests = _build_tests("test-classification")
        assert len(tests) == 1
        assert tests[0]["vars"]["text"] == "EC2 won't start"
        assert tests[0]["assert"][0]["type"] == "icontains"
        assert tests[0]["assert"][0]["value"] == "compute"

    def test_instruction_without_document(self, instruction_meta):
        from supereval.models import InstructionTestCase, InstructionInput, InstructionExpected
        from supereval.storage import append_cases
        cases = [
            InstructionTestCase(
                input=InstructionInput(instruction="Summarize this"),
                expected=InstructionExpected(rubric="Must be concise"),
            )
        ]
        append_cases("test-instruction", cases)
        tests = _build_tests("test-instruction")
        assert "instruction" in tests[0]["vars"]
        assert "document" not in tests[0]["vars"]
        assert tests[0]["assert"][0]["type"] == "llm-rubric"

    def test_instruction_with_document(self, instruction_meta):
        from supereval.models import InstructionTestCase, InstructionInput, InstructionExpected
        from supereval.storage import append_cases
        cases = [
            InstructionTestCase(
                input=InstructionInput(instruction="Summarize", document="Some text"),
                expected=InstructionExpected(rubric="Short"),
            )
        ]
        append_cases("test-instruction", cases)
        tests = _build_tests("test-instruction")
        assert tests[0]["vars"]["document"] == "Some text"

    def test_empty_dataset_returns_empty_list(self, qa_meta):
        tests = _build_tests("test-qa")
        assert tests == []


class TestExportToPromptfoo:
    def test_returns_valid_yaml(self, qa_dataset_with_cases):
        result = export_to_promptfoo("test-qa")
        parsed = yaml.safe_load(result)
        assert "tests" in parsed
        assert len(parsed["tests"]) == 3

    def test_description_included(self, qa_dataset_with_cases):
        result = export_to_promptfoo("test-qa")
        parsed = yaml.safe_load(result)
        assert parsed["tests"][0]["description"] == "SQS size limit"
