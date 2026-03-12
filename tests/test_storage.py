"""Tests for dataset storage (read/write operations)."""
from __future__ import annotations

import json

import pytest

from supereval.models import DatasetMeta, DatasetType, QAInput, QAExpected, QATestCase
from supereval.storage import (
    append_cases,
    dataset_path,
    datasets_dir,
    list_datasets,
    load_cases,
    load_dataset_meta,
    save_dataset_meta,
)


class TestDatasetsDir:
    def test_uses_env_var(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom"
        custom.mkdir()
        monkeypatch.setenv("SUPEREVAL_DATASETS_DIR", str(custom))
        assert datasets_dir() == custom

    def test_defaults_to_cwd_datasets(self, monkeypatch, tmp_path):
        monkeypatch.delenv("SUPEREVAL_DATASETS_DIR", raising=False)
        monkeypatch.chdir(tmp_path)
        assert datasets_dir() == tmp_path / "datasets"


class TestSaveLoadDatasetMeta:
    def test_roundtrip(self, datasets_dir):
        meta = DatasetMeta(name="my-ds", type=DatasetType.qa, description="Test", tags=["a"])
        save_dataset_meta(meta)
        loaded = load_dataset_meta("my-ds")
        assert loaded.id == meta.id
        assert loaded.name == "my-ds"
        assert loaded.tags == ["a"]
        assert loaded.type == DatasetType.qa

    def test_creates_directory(self, datasets_dir):
        meta = DatasetMeta(name="new-ds", type=DatasetType.classification)
        save_dataset_meta(meta)
        assert (datasets_dir / "new-ds" / "dataset.json").exists()

    def test_missing_dataset_raises(self, datasets_dir):
        with pytest.raises(FileNotFoundError, match="nonexistent"):
            load_dataset_meta("nonexistent")

    def test_thresholds_persisted(self, datasets_dir):
        from supereval.models import Thresholds
        meta = DatasetMeta(
            name="thresh-ds",
            type=DatasetType.qa,
            thresholds=Thresholds(pass_rate=0.85, fail_on_regression=False),
        )
        save_dataset_meta(meta)
        loaded = load_dataset_meta("thresh-ds")
        assert loaded.thresholds.pass_rate == 0.85
        assert loaded.thresholds.fail_on_regression is False


class TestAppendLoadCases:
    def test_append_and_load(self, qa_meta, qa_cases):
        append_cases("test-qa", qa_cases)
        loaded = load_cases("test-qa")
        assert len(loaded) == 3
        assert loaded[0].input.query == "What is the max SQS message size?"

    def test_append_is_additive(self, qa_meta, qa_cases):
        append_cases("test-qa", qa_cases[:1])
        append_cases("test-qa", qa_cases[1:])
        loaded = load_cases("test-qa")
        assert len(loaded) == 3

    def test_empty_dataset_returns_empty_list(self, qa_meta):
        assert load_cases("test-qa") == []

    def test_invalid_jsonl_raises(self, qa_meta, datasets_dir):
        cases_file = datasets_dir / "test-qa" / "cases.jsonl"
        cases_file.write_text('{"id":"tc_1","input":{"query":"q"},"expected":{"ground_truth":"a"}}\n')
        cases_file.write_text(cases_file.read_text() + "NOT VALID JSON\n")
        with pytest.raises(ValueError, match="line 2"):
            load_cases("test-qa")

    def test_cases_preserve_metadata(self, qa_meta):
        case = QATestCase(
            description="Test case",
            input=QAInput(query="q?"),
            expected=QAExpected(ground_truth="a"),
            tags=["tag1", "tag2"],
            difficulty="hard",
        )
        original_id = case.id
        append_cases("test-qa", [case])
        loaded = load_cases("test-qa")
        assert loaded[0].id == original_id
        assert loaded[0].tags == ["tag1", "tag2"]
        assert loaded[0].difficulty == "hard"


class TestListDatasets:
    def test_empty_dir_returns_empty(self, datasets_dir):
        assert list_datasets() == []

    def test_lists_all_datasets(self, qa_meta, classification_meta):
        results = list_datasets()
        names = [d.name for d in results]
        assert "test-qa" in names
        assert "test-classification" in names

    def test_ignores_non_dataset_dirs(self, datasets_dir):
        (datasets_dir / "not-a-dataset").mkdir()
        results = list_datasets()
        names = [d.name for d in results]
        assert "not-a-dataset" not in names

    def test_nonexistent_base_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SUPEREVAL_DATASETS_DIR", str(tmp_path / "nonexistent"))
        assert list_datasets() == []
