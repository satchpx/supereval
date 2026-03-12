"""Tests for document sources (LocalFileSource, S3Source)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from supereval.sources import Document, DocumentSource, LocalFileSource, S3Source


class TestDocumentProtocol:
    def test_local_file_source_satisfies_protocol(self, tmp_path):
        (tmp_path / "test.md").write_text("hello")
        source = LocalFileSource(tmp_path / "test.md")
        assert isinstance(source, DocumentSource)


class TestLocalFileSource:
    def test_load_txt_file(self, tmp_path):
        f = tmp_path / "guide.txt"
        f.write_text("Lambda max timeout is 15 minutes.")
        docs = LocalFileSource(f).load()
        assert len(docs) == 1
        assert docs[0].content == "Lambda max timeout is 15 minutes."
        assert docs[0].filename == "guide.txt"
        assert str(f) in docs[0].source

    def test_load_md_file(self, tmp_path):
        f = tmp_path / "readme.md"
        f.write_text("# S3 Guide\n\nS3 supports versioning.")
        docs = LocalFileSource(f).load()
        assert len(docs) == 1
        assert "versioning" in docs[0].content

    def test_load_directory(self, doc_dir):
        docs = LocalFileSource(doc_dir).load()
        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert "s3-guide.md" in filenames
        assert "lambda-guide.txt" in filenames

    def test_directory_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("top")
        (sub / "nested.txt").write_text("nested")
        docs = LocalFileSource(tmp_path, recursive=True).load()
        assert len(docs) == 2

    def test_directory_non_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.md").write_text("top")
        (sub / "nested.txt").write_text("nested")
        docs = LocalFileSource(tmp_path, recursive=False).load()
        assert len(docs) == 1
        assert docs[0].filename == "top.md"

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            LocalFileSource(tmp_path / "nonexistent.txt").load()

    def test_empty_directory_raises(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No supported files"):
            LocalFileSource(empty).load()

    def test_unsupported_extension_ignored_in_directory(self, tmp_path):
        (tmp_path / "doc.md").write_text("content")
        (tmp_path / "image.png").write_bytes(b"\x89PNG")
        docs = LocalFileSource(tmp_path).load()
        assert len(docs) == 1
        assert docs[0].filename == "doc.md"

    def test_unsupported_extension_as_single_file_raises(self, tmp_path):
        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        with pytest.raises(ValueError, match="Unsupported file type"):
            LocalFileSource(f).load()

    def test_pdf_raises_without_pypdf(self, tmp_path, monkeypatch):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "pypdf":
                raise ImportError("No module named 'pypdf'")
            return real_import(name, *args, **kwargs)

        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")
        monkeypatch.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="pypdf"):
            LocalFileSource(f).load()

    def test_content_stripped(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("\n\n  Hello world  \n\n")
        docs = LocalFileSource(f).load()
        assert docs[0].content == "Hello world"


# ---------------------------------------------------------------------------
# S3Source
# ---------------------------------------------------------------------------

def _make_s3_client(objects: dict[str, bytes]) -> MagicMock:
    """Build a mock boto3 S3 client with the given key→content mapping."""
    client = MagicMock()

    # list_objects_v2 paginator
    paginator = MagicMock()
    contents = [{"Key": k} for k in objects]
    paginator.paginate.return_value = [{"Contents": contents}]
    client.get_paginator.return_value = paginator

    # get_object
    def get_object(Bucket, Key):
        body = MagicMock()
        body.read.return_value = objects[Key]
        return {"Body": body}

    client.get_object.side_effect = get_object
    return client


class TestS3Source:
    def test_rejects_non_s3_uri(self):
        with pytest.raises(ValueError, match="s3://"):
            S3Source("not-an-s3-uri")

    def test_parses_bucket_and_prefix(self):
        s = S3Source("s3://my-bucket/docs/kb/")
        assert s.bucket == "my-bucket"
        assert s.prefix == "docs/kb/"

    def test_parses_bucket_only(self):
        s = S3Source("s3://my-bucket")
        assert s.bucket == "my-bucket"
        assert s.prefix == ""

    def test_satisfies_protocol(self):
        s = S3Source("s3://my-bucket/prefix/")
        assert isinstance(s, DocumentSource)

    def test_loads_txt_and_md(self):
        objects = {
            "docs/guide.txt": b"Lambda timeout is 15 minutes.",
            "docs/s3.md": b"# S3\n\nS3 supports versioning.",
        }
        with patch("boto3.client", return_value=_make_s3_client(objects)):
            docs = S3Source("s3://my-bucket/docs/").load()

        assert len(docs) == 2
        filenames = {d.filename for d in docs}
        assert "guide.txt" in filenames
        assert "s3.md" in filenames

    def test_document_source_field_is_s3_uri(self):
        objects = {"prefix/file.txt": b"content"}
        with patch("boto3.client", return_value=_make_s3_client(objects)):
            docs = S3Source("s3://my-bucket/prefix/").load()
        assert docs[0].source == "s3://my-bucket/prefix/file.txt"

    def test_skips_unsupported_extensions(self):
        objects = {
            "docs/guide.txt": b"content",
            "docs/image.png": b"\x89PNG",
        }
        with patch("boto3.client", return_value=_make_s3_client(objects)):
            docs = S3Source("s3://my-bucket/docs/").load()
        assert len(docs) == 1
        assert docs[0].filename == "guide.txt"

    def test_empty_prefix_raises(self):
        with patch("boto3.client", return_value=_make_s3_client({})):
            with pytest.raises(ValueError, match="No supported files"):
                S3Source("s3://my-bucket/empty/").load()

    def test_boto3_import_error_raises_clearly(self):
        with patch.dict("sys.modules", {"boto3": None}):
            s = S3Source("s3://bucket/prefix/")
            s._client = None
            with pytest.raises((ImportError, TypeError)):
                _ = s.client
