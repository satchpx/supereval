"""Tests for Pydantic models and schema validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from supereval.models import (
    ClassificationExpected,
    ClassificationInput,
    ClassificationTestCase,
    DatasetMeta,
    DatasetType,
    InstructionExpected,
    InstructionInput,
    InstructionTestCase,
    QAExpected,
    QAInput,
    QATestCase,
    Thresholds,
    TYPE_TO_MODEL,
)


class TestThresholds:
    def test_defaults(self):
        t = Thresholds()
        assert t.pass_rate == 1.0
        assert t.fail_on_regression is True
        assert t.max_score_drop is None

    def test_custom_values(self):
        t = Thresholds(pass_rate=0.9, fail_on_regression=False, max_score_drop=0.05)
        assert t.pass_rate == 0.9
        assert t.fail_on_regression is False
        assert t.max_score_drop == 0.05


class TestDatasetMeta:
    def test_defaults(self):
        meta = DatasetMeta(name="my-dataset", type=DatasetType.qa)
        assert meta.id.startswith("ds_")
        assert meta.description == ""
        assert meta.tags == []
        assert meta.labels is None
        assert isinstance(meta.thresholds, Thresholds)

    def test_with_all_fields(self):
        meta = DatasetMeta(
            name="my-dataset",
            type=DatasetType.classification,
            description="Test",
            tags=["aws"],
            labels=["a", "b"],
            author="tester",
        )
        assert meta.labels == ["a", "b"]
        assert meta.author == "tester"

    def test_roundtrip_json(self):
        meta = DatasetMeta(name="roundtrip", type=DatasetType.qa, tags=["test"])
        restored = DatasetMeta.model_validate_json(meta.model_dump_json())
        assert restored.name == meta.name
        assert restored.id == meta.id
        assert restored.tags == meta.tags


class TestQATestCase:
    def test_valid(self):
        case = QATestCase(
            input=QAInput(query="What is S3?"),
            expected=QAExpected(ground_truth="S3 is object storage."),
        )
        assert case.id.startswith("tc_")
        assert case.difficulty is None
        assert case.tags == []

    def test_missing_query_raises(self):
        with pytest.raises(ValidationError):
            QATestCase(input={}, expected=QAExpected(ground_truth="answer"))

    def test_difficulty_enum(self):
        case = QATestCase(
            input=QAInput(query="q"),
            expected=QAExpected(ground_truth="a"),
            difficulty="hard",
        )
        assert case.difficulty == "hard"

    def test_invalid_difficulty_raises(self):
        with pytest.raises(ValidationError):
            QATestCase(
                input=QAInput(query="q"),
                expected=QAExpected(ground_truth="a"),
                difficulty="impossible",
            )

    def test_roundtrip_json(self):
        case = QATestCase(
            input=QAInput(query="q?"),
            expected=QAExpected(ground_truth="a"),
            tags=["tag1"],
            difficulty="easy",
        )
        restored = QATestCase.model_validate_json(case.model_dump_json())
        assert restored.input.query == "q?"
        assert restored.expected.ground_truth == "a"
        assert restored.id == case.id


class TestClassificationTestCase:
    def test_valid(self):
        case = ClassificationTestCase(
            input=ClassificationInput(text="My EC2 won't start"),
            expected=ClassificationExpected(label="compute"),
        )
        assert case.expected.label == "compute"

    def test_missing_text_raises(self):
        with pytest.raises(ValidationError):
            ClassificationTestCase(
                input={},
                expected=ClassificationExpected(label="compute"),
            )


class TestInstructionTestCase:
    def test_valid_without_document(self):
        case = InstructionTestCase(
            input=InstructionInput(instruction="Summarize this"),
            expected=InstructionExpected(rubric="Must be under 50 words"),
        )
        assert case.input.document is None

    def test_valid_with_document(self):
        case = InstructionTestCase(
            input=InstructionInput(instruction="Summarize", document="Long text here..."),
            expected=InstructionExpected(rubric="Short summary"),
        )
        assert case.input.document == "Long text here..."


class TestTypeToModel:
    def test_all_types_mapped(self):
        assert TYPE_TO_MODEL[DatasetType.qa] is QATestCase
        assert TYPE_TO_MODEL[DatasetType.classification] is ClassificationTestCase
        assert TYPE_TO_MODEL[DatasetType.instruction] is InstructionTestCase
