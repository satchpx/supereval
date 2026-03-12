from __future__ import annotations

import yaml

from .models import (
    ClassificationTestCase,
    DatasetType,
    InstructionTestCase,
    QATestCase,
)
from .storage import load_cases, load_dataset_meta


def _build_tests(name: str) -> list[dict]:
    """Return test cases as a Python list (Promptfoo format). Used by the runner."""
    meta = load_dataset_meta(name)
    cases = load_cases(name)
    tests = []

    for case in cases:
        if meta.type == DatasetType.qa:
            assert isinstance(case, QATestCase)
            tests.append({
                "description": case.description,
                "vars": {"query": case.input.query},
                "assert": [
                    {"type": "icontains", "value": case.expected.ground_truth},
                    {
                        "type": "llm-rubric",
                        "value": f"The answer must correctly address the following: {case.expected.ground_truth}",
                    },
                ],
            })

        elif meta.type == DatasetType.classification:
            assert isinstance(case, ClassificationTestCase)
            tests.append({
                "description": case.description,
                "vars": {"text": case.input.text},
                "assert": [
                    {"type": "icontains", "value": case.expected.label},
                ],
            })

        elif meta.type == DatasetType.instruction:
            assert isinstance(case, InstructionTestCase)
            vars_: dict = {"instruction": case.input.instruction}
            if case.input.document:
                vars_["document"] = case.input.document
            tests.append({
                "description": case.description,
                "vars": vars_,
                "assert": [
                    {"type": "llm-rubric", "value": case.expected.rubric},
                ],
            })

    return tests


def export_to_promptfoo(name: str) -> str:
    tests = _build_tests(name)
    return yaml.dump(
        {"tests": tests},
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
