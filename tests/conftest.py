"""Shared fixtures for all supereval tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from supereval.models import (
    ClassificationTestCase,
    ClassificationInput,
    ClassificationExpected,
    DatasetMeta,
    DatasetType,
    InstructionTestCase,
    InstructionInput,
    InstructionExpected,
    QATestCase,
    QAInput,
    QAExpected,
)
from supereval.runner import CaseResult, RunResult
from supereval.storage import append_cases, save_dataset_meta


@pytest.fixture
def datasets_dir(tmp_path, monkeypatch):
    """Isolated datasets directory; injects SUPEREVAL_DATASETS_DIR so all storage
    calls resolve to a temp path instead of ./datasets."""
    d = tmp_path / "datasets"
    d.mkdir()
    monkeypatch.setenv("SUPEREVAL_DATASETS_DIR", str(d))
    return d


@pytest.fixture
def qa_meta(datasets_dir):
    meta = DatasetMeta(
        name="test-qa",
        type=DatasetType.qa,
        description="Test QA dataset",
        tags=["aws", "test"],
    )
    save_dataset_meta(meta)
    return meta


@pytest.fixture
def classification_meta(datasets_dir):
    meta = DatasetMeta(
        name="test-classification",
        type=DatasetType.classification,
        description="Test classification dataset",
        labels=["networking", "compute", "storage"],
    )
    save_dataset_meta(meta)
    return meta


@pytest.fixture
def instruction_meta(datasets_dir):
    meta = DatasetMeta(
        name="test-instruction",
        type=DatasetType.instruction,
        description="Test instruction dataset",
    )
    save_dataset_meta(meta)
    return meta


@pytest.fixture
def qa_cases():
    return [
        QATestCase(
            description="SQS size limit",
            input=QAInput(query="What is the max SQS message size?"),
            expected=QAExpected(ground_truth="256 KB"),
            tags=["sqs", "limits"],
            difficulty="easy",
        ),
        QATestCase(
            description="Lambda timeout",
            input=QAInput(query="What is the max Lambda timeout?"),
            expected=QAExpected(ground_truth="15 minutes"),
            tags=["lambda", "limits"],
            difficulty="easy",
        ),
        QATestCase(
            description="S3 versioning",
            input=QAInput(query="Does S3 versioning protect against accidental deletes?"),
            expected=QAExpected(ground_truth="Yes, deletes place a delete marker"),
            tags=["s3"],
            difficulty="medium",
        ),
    ]


@pytest.fixture
def qa_dataset_with_cases(qa_meta, qa_cases):
    append_cases("test-qa", qa_cases)
    return qa_meta


@pytest.fixture
def sample_run_result():
    return RunResult(
        dataset="test-qa",
        providers=["anthropic:claude-opus-4-6"],
        cases=[
            CaseResult(vars={"query": "What is the max SQS message size?"}, passed=True, score=1.0, latency_ms=200),
            CaseResult(vars={"query": "What is the max Lambda timeout?"}, passed=True, score=1.0, latency_ms=180),
            CaseResult(vars={"query": "Does S3 versioning protect against accidental deletes?"}, passed=False, score=0.2, latency_ms=220),
        ],
        run_id="run_test001",
    )


@pytest.fixture
def agent_meta(datasets_dir):
    from supereval.agent.models import AgentDatasetMeta, ToolSpec
    from supereval.agent.storage import save_agent_dataset_meta
    meta = AgentDatasetMeta(
        name="test-agent",
        description="Test agent dataset",
        tools=[
            ToolSpec(
                name="search",
                description="Search the knowledge base",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}},
            ),
            ToolSpec(
                name="get_item",
                description="Get a specific item by ID",
                parameters={"type": "object", "properties": {"id": {"type": "string"}}},
            ),
        ],
    )
    save_agent_dataset_meta(meta)
    return meta


@pytest.fixture
def sample_agent_cases():
    from supereval.agent.models import (
        AgentTestCase, AgentTask, AgentExpected, MockResponse, MustCallWith,
    )
    return [
        AgentTestCase(
            id="atc_case001",
            description="Find item region",
            input=AgentTask(task="Which region is item-42 in?"),
            tools={
                "search": MockResponse(
                    response=[{"id": "item-42", "region": "us-west-2"}]
                ),
                "get_item": MockResponse(
                    response={"id": "item-42", "region": "us-west-2"}
                ),
            },
            expected=AgentExpected(
                answer="us-west-2",
                answer_match="contains",
                must_call=["search"],
                max_steps=6,
                max_tool_calls=3,
            ),
        ),
        AgentTestCase(
            id="atc_case002",
            description="Handle access denied gracefully",
            input=AgentTask(task="Get item-99"),
            tools={
                "search": MockResponse(response=[]),
                "get_item": MockResponse(error="AccessDenied"),
            },
            expected=AgentExpected(
                answer="not found",
                answer_match="contains",
                must_not_call=["delete_item"],
            ),
        ),
    ]


@pytest.fixture
def agent_dataset_with_cases(agent_meta, sample_agent_cases):
    from supereval.agent.storage import append_agent_cases
    append_agent_cases("test-agent", sample_agent_cases)
    return agent_meta


@pytest.fixture
def doc_dir(tmp_path):
    """A temporary directory with sample document files."""
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "s3-guide.md").write_text(
        "# S3 User Guide\n\nS3 supports versioning. "
        "Versioning protects against accidental deletion.\n\n"
        "Maximum object size is 5 TB.\n"
    )
    (docs / "lambda-guide.txt").write_text(
        "Lambda functions have a maximum timeout of 15 minutes.\n"
        "Memory can be configured from 128 MB to 10 GB.\n"
    )
    return docs
