"""Tests for RAG dataset storage — save/load/list/append."""
from __future__ import annotations

import pytest
from supereval.rag.models import (
    RagDatasetMeta,
    RagExpected,
    RagInput,
    RagTestCase,
)
from supereval.rag.storage import (
    append_rag_cases,
    list_rag_datasets,
    load_rag_cases,
    load_rag_dataset_meta,
    rag_dataset_path,
    save_rag_dataset_meta,
)


@pytest.fixture
def rag_meta(datasets_dir):
    meta = RagDatasetMeta(name="test-rag", description="RAG test")
    save_rag_dataset_meta(meta)
    return meta


@pytest.fixture
def sample_rag_cases():
    return [
        RagTestCase(
            id="rtc_001",
            description="Region lookup",
            input=RagInput(
                query="What region is my-bucket in?",
                retrieved_contexts=["my-bucket is in us-west-2."],
            ),
            expected=RagExpected(ground_truth="us-west-2"),
            tags=["s3"],
            difficulty="easy",
        ),
        RagTestCase(
            id="rtc_002",
            description="Lambda timeout",
            input=RagInput(
                query="What is the max Lambda timeout?",
                retrieved_contexts=["Lambda functions have a max timeout of 15 minutes."],
            ),
            expected=RagExpected(ground_truth="15 minutes"),
            tags=["lambda"],
        ),
    ]


class TestSaveLoadMeta:
    def test_save_and_load(self, rag_meta):
        loaded = load_rag_dataset_meta("test-rag")
        assert loaded.name == "test-rag"
        assert loaded.description == "RAG test"

    def test_creates_directory(self, datasets_dir, rag_meta):
        assert rag_dataset_path("test-rag").exists()

    def test_load_missing_raises(self, datasets_dir):
        with pytest.raises(FileNotFoundError, match="test-missing"):
            load_rag_dataset_meta("test-missing")


class TestAppendLoadCases:
    def test_append_and_load(self, rag_meta, sample_rag_cases):
        append_rag_cases("test-rag", sample_rag_cases)
        loaded = load_rag_cases("test-rag")
        assert len(loaded) == 2
        assert loaded[0].id == "rtc_001"
        assert loaded[1].id == "rtc_002"

    def test_append_is_additive(self, rag_meta, sample_rag_cases):
        append_rag_cases("test-rag", [sample_rag_cases[0]])
        append_rag_cases("test-rag", [sample_rag_cases[1]])
        loaded = load_rag_cases("test-rag")
        assert len(loaded) == 2

    def test_load_empty_returns_empty_list(self, rag_meta):
        assert load_rag_cases("test-rag") == []

    def test_contexts_preserved(self, rag_meta, sample_rag_cases):
        append_rag_cases("test-rag", sample_rag_cases)
        loaded = load_rag_cases("test-rag")
        assert loaded[0].input.retrieved_contexts == ["my-bucket is in us-west-2."]

    def test_invalid_line_raises(self, rag_meta, datasets_dir):
        path = rag_dataset_path("test-rag") / "cases.jsonl"
        path.write_text('{"invalid": true}\n')
        with pytest.raises(ValueError, match="Validation errors"):
            load_rag_cases("test-rag")


class TestListDatasets:
    def test_lists_all_rag_datasets(self, datasets_dir):
        for name in ("rag-a", "rag-b"):
            save_rag_dataset_meta(RagDatasetMeta(name=name))
        datasets = list_rag_datasets()
        names = {d.name for d in datasets}
        assert "rag-a" in names
        assert "rag-b" in names

    def test_empty_when_no_datasets(self, datasets_dir):
        assert list_rag_datasets() == []

    def test_ignores_directories_without_meta(self, datasets_dir):
        (datasets_dir / "rag" / "ghost-dir").mkdir(parents=True)
        assert list_rag_datasets() == []
