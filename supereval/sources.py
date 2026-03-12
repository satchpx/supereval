"""
Document sources for synthetic test case generation.

Supported:  local files (.txt, .md, .pdf*), Amazon S3
Planned:    URLs, Amazon Kendra, Bedrock Knowledge Base (see ROADMAP.md)

* PDF requires: pip install supereval[pdf]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

SUPPORTED_EXTENSIONS = {".txt", ".md"}
OPTIONAL_EXTENSIONS = {".pdf"}  # requires pypdf
ALL_EXTENSIONS = SUPPORTED_EXTENSIONS | OPTIONAL_EXTENSIONS


@dataclass
class Document:
    content: str
    filename: str
    source: str  # human-readable origin (file path, S3 URI, URL, etc.)


@runtime_checkable
class DocumentSource(Protocol):
    """
    Protocol all document sources must implement.
    Satisfying this protocol is sufficient — no base class needed.
    """
    def load(self) -> list[Document]:
        ...


class LocalFileSource:
    """
    Load documents from a local file or directory.

    Supported formats : .txt, .md
    Optional formats  : .pdf  (pip install supereval[pdf])
    Not yet supported : S3, URLs — see ROADMAP.md
    """

    def __init__(self, path: Path | str, recursive: bool = True):
        self.path = Path(path)
        self.recursive = recursive

    def load(self) -> list[Document]:
        if self.path.is_file():
            return [self._load_file(self.path)]
        if self.path.is_dir():
            return self._load_directory()
        raise FileNotFoundError(f"Path not found: {self.path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_directory(self) -> list[Document]:
        pattern = "**/*" if self.recursive else "*"
        files = sorted(
            f for f in self.path.glob(pattern)
            if f.is_file() and f.suffix.lower() in ALL_EXTENSIONS
        )
        if not files:
            raise ValueError(
                f"No supported files found in {self.path}. "
                f"Supported: {', '.join(sorted(ALL_EXTENSIONS))}"
            )
        return [self._load_file(f) for f in files]

    def _load_file(self, path: Path) -> Document:
        ext = path.suffix.lower()
        if ext in (".txt", ".md"):
            content = path.read_text(encoding="utf-8", errors="replace")
        elif ext == ".pdf":
            content = self._load_pdf(path)
        else:
            raise ValueError(
                f"Unsupported file type: {ext}. "
                f"Supported: {', '.join(sorted(ALL_EXTENSIONS))}"
            )
        return Document(content=content.strip(), filename=path.name, source=str(path))

    @staticmethod
    def _load_pdf(path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PDF support requires pypdf. Install it with: pip install 'supereval[pdf]'"
            )
        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)


class S3Source:
    """
    Load documents from Amazon S3.

    URI format: s3://bucket-name/optional/prefix

    Requires boto3 and S3 read permissions (s3:ListObjectsV2, s3:GetObject).
    Supports .txt, .md, .pdf (PDF requires pip install 'supereval[pdf]').
    """

    def __init__(self, uri: str, region: str = "us-east-1"):
        if not uri.startswith("s3://"):
            raise ValueError(f"S3 URI must start with 's3://': {uri!r}")
        without_scheme = uri[5:]
        slash = without_scheme.find("/")
        if slash == -1:
            self.bucket = without_scheme
            self.prefix = ""
        else:
            self.bucket = without_scheme[:slash]
            self.prefix = without_scheme[slash + 1:]
        self.region = region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("s3", region_name=self.region)
            except ImportError:
                raise ImportError(
                    "boto3 is required for S3 sources. Install with: pip install boto3"
                )
        return self._client

    def load(self) -> list[Document]:
        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=self.prefix)
        keys: list[str] = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if Path(key).suffix.lower() in ALL_EXTENSIONS:
                    keys.append(key)
        if not keys:
            raise ValueError(
                f"No supported files found at s3://{self.bucket}/{self.prefix}. "
                f"Supported: {', '.join(sorted(ALL_EXTENSIONS))}"
            )
        return [self._load_key(key) for key in sorted(keys)]

    def _load_key(self, key: str) -> Document:
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        body: bytes = response["Body"].read()
        ext = Path(key).suffix.lower()
        if ext in (".txt", ".md"):
            content = body.decode("utf-8", errors="replace").strip()
        elif ext == ".pdf":
            content = self._parse_pdf(body)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        return Document(
            content=content,
            filename=Path(key).name,
            source=f"s3://{self.bucket}/{key}",
        )

    @staticmethod
    def _parse_pdf(data: bytes) -> str:
        try:
            import io
            from pypdf import PdfReader
        except ImportError:
            raise ImportError(
                "PDF support requires pypdf. Install with: pip install 'supereval[pdf]'"
            )
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
