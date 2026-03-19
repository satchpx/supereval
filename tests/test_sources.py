"""Tests for document sources (LocalFileSource, S3Source, URLSource)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from supereval.sources import Document, DocumentSource, LocalFileSource, S3Source, URLSource


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


# ---------------------------------------------------------------------------
# URLSource
# ---------------------------------------------------------------------------

def _mock_urlopen(content: bytes, content_type: str = "text/plain"):
    """Build a mock context manager returned by urllib.request.urlopen."""
    mock_resp = MagicMock()
    mock_resp.headers.get.return_value = content_type
    mock_resp.read.return_value = content
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestURLSource:
    def test_rejects_non_http_url(self):
        with pytest.raises(ValueError, match="http"):
            URLSource("ftp://example.com/file")

    def test_rejects_relative_url(self):
        with pytest.raises(ValueError, match="http"):
            URLSource("example.com/page")

    def test_satisfies_protocol(self):
        s = URLSource("https://example.com/doc.txt")
        assert isinstance(s, DocumentSource)

    def test_loads_plain_text(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(b"Hello world")):
            docs = URLSource("https://example.com/doc.txt").load()
        assert len(docs) == 1
        assert docs[0].content == "Hello world"
        assert docs[0].source == "https://example.com/doc.txt"

    def test_filename_derived_from_url(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(b"content")):
            docs = URLSource("https://example.com/guide.txt").load()
        assert docs[0].filename == "guide.txt"

    def test_filename_defaults_to_txt_when_no_extension(self):
        with patch("urllib.request.urlopen", return_value=_mock_urlopen(b"content")):
            docs = URLSource("https://example.com/page").load()
        assert docs[0].filename.endswith(".txt")

    def test_strips_html_tags(self):
        html = b"<html><body><h1>Title</h1><p>Some content here.</p></body></html>"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(html, content_type="text/html"),
        ):
            docs = URLSource("https://example.com/page.html").load()
        assert "<html>" not in docs[0].content
        assert "Title" in docs[0].content
        assert "Some content here." in docs[0].content

    def test_strips_script_and_style_blocks(self):
        html = b"<html><head><style>.cls{color:red}</style><script>alert(1)</script></head><body>Real content</body></html>"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(html, content_type="text/html"),
        ):
            docs = URLSource("https://example.com/page.html").load()
        assert "alert" not in docs[0].content
        assert ".cls" not in docs[0].content
        assert "Real content" in docs[0].content

    def test_decodes_html_entities(self):
        html = b"<p>a &amp; b &lt;c&gt; &quot;d&quot;</p>"
        with patch(
            "urllib.request.urlopen",
            return_value=_mock_urlopen(html, content_type="text/html"),
        ):
            docs = URLSource("https://example.com/page.html").load()
        assert "a & b <c>" in docs[0].content
        assert '"d"' in docs[0].content

    def test_multiple_urls_loads_all(self):
        responses = [
            _mock_urlopen(b"First page"),
            _mock_urlopen(b"Second page"),
        ]
        with patch("urllib.request.urlopen", side_effect=responses):
            docs = URLSource([
                "https://example.com/a.txt",
                "https://example.com/b.txt",
            ]).load()
        assert len(docs) == 2
        assert docs[0].content == "First page"
        assert docs[1].content == "Second page"

    def test_raw_html_without_content_type_still_stripped(self):
        """HTML detected by content even when Content-Type is missing."""
        html = b"<html><body>Page content</body></html>"
        mock_resp = _mock_urlopen(html, content_type="")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            docs = URLSource("https://example.com/page").load()
        assert "<html>" not in docs[0].content
        assert "Page content" in docs[0].content

    def test_crawl_multiple_urls_raises(self):
        with pytest.raises(ValueError, match="single seed"):
            URLSource(["https://a.com/", "https://b.com/"], crawl=True)


class TestURLSourceCrawl:
    """Tests for URLSource crawl mode."""

    def _make_responses(self, url_to_content: dict) -> callable:
        """Return a urlopen side_effect that dispatches by URL."""
        def fake_urlopen(req, timeout=None):
            url = req.get_full_url()
            content, ctype = url_to_content.get(url, (b"<html><body>fallback</body></html>", "text/html"))
            return _mock_urlopen(content, ctype)
        return fake_urlopen

    def test_crawl_fetches_seed_page(self):
        pages = {
            "https://example.com/docs/": (b"<html><body>Root page</body></html>", "text/html"),
        }
        with patch("urllib.request.urlopen", side_effect=self._make_responses(pages)):
            docs = URLSource("https://example.com/docs/", crawl=True).load()
        assert len(docs) >= 1
        assert "Root page" in docs[0].content

    def test_crawl_follows_links_under_prefix(self):
        root = b'<html><body><a href="/docs/page1.html">P1</a><a href="/docs/page2.html">P2</a>Root</body></html>'
        pages = {
            "https://example.com/docs/": (root, "text/html"),
            "https://example.com/docs/page1.html": (b"<html><body>Page one</body></html>", "text/html"),
            "https://example.com/docs/page2.html": (b"<html><body>Page two</body></html>", "text/html"),
        }
        with patch("urllib.request.urlopen", side_effect=self._make_responses(pages)):
            docs = URLSource("https://example.com/docs/", crawl=True, max_pages=10).load()
        sources = {d.source for d in docs}
        assert "https://example.com/docs/" in sources
        assert "https://example.com/docs/page1.html" in sources
        assert "https://example.com/docs/page2.html" in sources

    def test_crawl_does_not_follow_external_links(self):
        root = b'<html><body><a href="https://other.com/page">external</a><a href="/docs/internal.html">internal</a>Root</body></html>'
        pages = {
            "https://example.com/docs/": (root, "text/html"),
            "https://example.com/docs/internal.html": (b"<html><body>Internal</body></html>", "text/html"),
        }
        with patch("urllib.request.urlopen", side_effect=self._make_responses(pages)):
            docs = URLSource("https://example.com/docs/", crawl=True, max_pages=10).load()
        sources = {d.source for d in docs}
        assert "https://other.com/page" not in sources
        assert "https://example.com/docs/internal.html" in sources

    def test_crawl_does_not_follow_parent_directory_links(self):
        """Links going above the seed prefix are not followed."""
        root = b'<html><body><a href="/other-section/page.html">up</a><a href="/docs/sub.html">sub</a>Root</body></html>'
        pages = {
            "https://example.com/docs/": (root, "text/html"),
            "https://example.com/docs/sub.html": (b"<html><body>Sub</body></html>", "text/html"),
        }
        with patch("urllib.request.urlopen", side_effect=self._make_responses(pages)):
            docs = URLSource("https://example.com/docs/", crawl=True, max_pages=10).load()
        sources = {d.source for d in docs}
        assert "https://example.com/other-section/page.html" not in sources

    def test_crawl_respects_max_pages(self):
        # Root links to 5 sub-pages but max_pages=3
        links = "".join(f'<a href="/docs/p{i}.html">p{i}</a>' for i in range(5))
        root = f"<html><body>{links}Root</body></html>".encode()
        pages: dict = {"https://example.com/docs/": (root, "text/html")}
        for i in range(5):
            pages[f"https://example.com/docs/p{i}.html"] = (
                f"<html><body>Page {i}</body></html>".encode(), "text/html"
            )
        with patch("urllib.request.urlopen", side_effect=self._make_responses(pages)):
            docs = URLSource("https://example.com/docs/", crawl=True, max_pages=3).load()
        assert len(docs) == 3

    def test_crawl_does_not_revisit_pages(self):
        # Two pages both link back to root — should not cause infinite loop or duplicates
        root = b'<html><body><a href="/docs/a.html">a</a>Root</body></html>'
        page_a = b'<html><body><a href="/docs/">back</a>Page A</body></html>'
        pages = {
            "https://example.com/docs/": (root, "text/html"),
            "https://example.com/docs/a.html": (page_a, "text/html"),
        }
        with patch("urllib.request.urlopen", side_effect=self._make_responses(pages)):
            docs = URLSource("https://example.com/docs/", crawl=True, max_pages=10).load()
        sources = [d.source for d in docs]
        assert sources.count("https://example.com/docs/") == 1

    def test_crawl_skips_failing_pages(self):
        """Pages that return errors are silently skipped; crawl continues."""
        import urllib.error
        root = b'<html><body><a href="/docs/good.html">good</a><a href="/docs/bad.html">bad</a>Root</body></html>'

        def fake_urlopen(req, timeout=None):
            url = req.get_full_url()
            if "bad" in url:
                raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)
            if "good" in url:
                return _mock_urlopen(b"<html><body>Good page</body></html>", "text/html")
            return _mock_urlopen(root, "text/html")

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            docs = URLSource("https://example.com/docs/", crawl=True, max_pages=10).load()
        sources = {d.source for d in docs}
        assert "https://example.com/docs/good.html" in sources
        assert "https://example.com/docs/bad.html" not in sources

    def test_extract_links_resolves_relative_urls(self):
        html = '<a href="page.html">rel</a><a href="/abs/path.html">abs</a>'
        links = URLSource._extract_links(html, "https://example.com/docs/index.html")
        assert "https://example.com/docs/page.html" in links
        assert "https://example.com/abs/path.html" in links

    def test_extract_links_strips_fragments(self):
        html = '<a href="page.html#section">link</a>'
        links = URLSource._extract_links(html, "https://example.com/docs/")
        assert "https://example.com/docs/page.html" in links
        assert "#section" not in links[0]

    def test_extract_links_deduplicates(self):
        html = '<a href="page.html">1</a><a href="page.html">2</a>'
        links = URLSource._extract_links(html, "https://example.com/docs/")
        assert links.count("https://example.com/docs/page.html") == 1
