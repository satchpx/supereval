"""Tests for dataset versioning — tag, list, load, restore."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from supereval.versioning import (
    VersionInfo,
    list_versions,
    load_version_cases,
    restore_version,
    tag_version,
    validate_version,
    version_exists,
)


# ---------------------------------------------------------------------------
# validate_version
# ---------------------------------------------------------------------------

class TestValidateVersion:
    @pytest.mark.parametrize("v", ["v1.0.0", "v0.1.0", "v10.20.300"])
    def test_valid_versions(self, v):
        validate_version(v)  # should not raise

    @pytest.mark.parametrize("v", ["1.0.0", "v1.0", "v1.0.0.0", "latest", "", "V1.0.0"])
    def test_invalid_versions(self, v):
        with pytest.raises(ValueError, match="Invalid version"):
            validate_version(v)


# ---------------------------------------------------------------------------
# tag_version
# ---------------------------------------------------------------------------

class TestTagVersion:
    def test_creates_version_directory(self, qa_meta, qa_cases, datasets_dir):
        tag_version("test-qa", "v1.0.0")
        vdir = datasets_dir / "test-qa" / "versions" / "v1.0.0"
        assert vdir.is_dir()

    def test_snapshots_cases(self, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        tag_version("test-qa", "v1.0.0")
        snapshot = datasets_dir / "test-qa" / "versions" / "v1.0.0" / "cases.jsonl"
        assert snapshot.exists()
        lines = [l for l in snapshot.read_text().splitlines() if l.strip()]
        assert len(lines) == len(qa_cases)

    def test_snapshots_dataset_json(self, qa_meta, datasets_dir):
        tag_version("test-qa", "v1.0.0")
        snapshot = datasets_dir / "test-qa" / "versions" / "v1.0.0" / "dataset.json"
        assert snapshot.exists()

    def test_writes_version_json(self, qa_meta, datasets_dir):
        info = tag_version("test-qa", "v1.0.0", description="first release")
        vinfo = datasets_dir / "test-qa" / "versions" / "v1.0.0" / "version.json"
        assert vinfo.exists()
        d = json.loads(vinfo.read_text())
        assert d["version"] == "v1.0.0"
        assert d["description"] == "first release"
        assert "created_at" in d

    def test_returns_version_info(self, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        info = tag_version("test-qa", "v1.0.0")
        assert isinstance(info, VersionInfo)
        assert info.version == "v1.0.0"
        assert info.case_count == len(qa_cases)

    def test_case_count_zero_when_no_cases(self, qa_meta):
        info = tag_version("test-qa", "v1.0.0")
        assert info.case_count == 0

    def test_duplicate_version_raises(self, qa_meta):
        tag_version("test-qa", "v1.0.0")
        with pytest.raises(ValueError, match="already exists"):
            tag_version("test-qa", "v1.0.0")

    def test_invalid_version_format_raises(self, qa_meta):
        with pytest.raises(ValueError, match="Invalid version"):
            tag_version("test-qa", "1.0.0")

    def test_missing_dataset_raises(self):
        with pytest.raises(FileNotFoundError):
            tag_version("nonexistent", "v1.0.0")

    def test_multiple_versions_coexist(self, qa_meta):
        tag_version("test-qa", "v1.0.0")
        tag_version("test-qa", "v1.1.0")
        assert version_exists("test-qa", "v1.0.0")
        assert version_exists("test-qa", "v1.1.0")


# ---------------------------------------------------------------------------
# list_versions
# ---------------------------------------------------------------------------

class TestListVersions:
    def test_empty_when_no_versions(self, qa_meta):
        assert list_versions("test-qa") == []

    def test_returns_tagged_versions(self, qa_meta):
        tag_version("test-qa", "v1.0.0", description="first")
        tag_version("test-qa", "v1.1.0", description="second")
        versions = list_versions("test-qa")
        assert len(versions) == 2
        assert versions[0].version == "v1.0.0"
        assert versions[1].version == "v1.1.0"

    def test_sorted_by_version_string(self, qa_meta):
        tag_version("test-qa", "v1.1.0")
        tag_version("test-qa", "v1.0.0")
        versions = list_versions("test-qa")
        assert [v.version for v in versions] == ["v1.0.0", "v1.1.0"]

    def test_description_preserved(self, qa_meta):
        tag_version("test-qa", "v1.0.0", description="release note")
        versions = list_versions("test-qa")
        assert versions[0].description == "release note"


# ---------------------------------------------------------------------------
# load_version_cases
# ---------------------------------------------------------------------------

class TestLoadVersionCases:
    def test_loads_cases_from_version(self, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        tag_version("test-qa", "v1.0.0")
        cases = load_version_cases("test-qa", "v1.0.0")
        assert len(cases) == len(qa_cases)

    def test_returns_empty_list_for_zero_cases(self, qa_meta):
        tag_version("test-qa", "v1.0.0")
        cases = load_version_cases("test-qa", "v1.0.0")
        assert cases == []

    def test_raises_for_missing_version(self, qa_meta):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_version_cases("test-qa", "v9.9.9")

    def test_case_content_preserved(self, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        tag_version("test-qa", "v1.0.0")
        cases = load_version_cases("test-qa", "v1.0.0")
        assert cases[0].input.query == qa_cases[0].input.query


# ---------------------------------------------------------------------------
# restore_version
# ---------------------------------------------------------------------------

class TestRestoreVersion:
    def test_restores_cases(self, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases, load_cases
        append_cases("test-qa", qa_cases)
        tag_version("test-qa", "v1.0.0")

        # Add an extra case after tagging
        append_cases("test-qa", [qa_cases[0]])
        assert len(load_cases("test-qa")) == len(qa_cases) + 1

        # Restore to v1.0.0 — should drop the extra case
        count = restore_version("test-qa", "v1.0.0")
        assert count == len(qa_cases)
        assert len(load_cases("test-qa")) == len(qa_cases)

    def test_returns_case_count(self, qa_meta, qa_cases, datasets_dir):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        tag_version("test-qa", "v1.0.0")
        count = restore_version("test-qa", "v1.0.0")
        assert count == len(qa_cases)

    def test_raises_for_missing_version(self, qa_meta):
        with pytest.raises(FileNotFoundError, match="not found"):
            restore_version("test-qa", "v9.9.9")

    def test_version_snapshot_not_modified_after_restore(
        self, qa_meta, qa_cases, datasets_dir
    ):
        from supereval.storage import append_cases
        append_cases("test-qa", qa_cases)
        tag_version("test-qa", "v1.0.0")
        restore_version("test-qa", "v1.0.0")

        # Snapshot should still be intact after restore
        snapshot_cases = load_version_cases("test-qa", "v1.0.0")
        assert len(snapshot_cases) == len(qa_cases)
