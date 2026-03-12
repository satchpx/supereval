# Troubleshooting

## Installation

### `promptfoo: command not found`

```
RuntimeError: promptfoo not found. Install it with: npm install -g promptfoo
```

**Fix:**
```bash
npm install -g promptfoo
promptfoo --version  # verify
```

If you don't have npm, install Node.js first: [nodejs.org](https://nodejs.org)

If you can't install globally, `npx promptfoo@latest` will be used automatically as a fallback.

---

### `ModuleNotFoundError: No module named 'supereval'`

You ran `supereval` before installing the package, or you're in the wrong virtual environment.

**Fix:**
```bash
pip install -e .
```

---

### `ImportError: No module named 'pypdf'`

You're trying to generate from a PDF without the optional dependency.

**Fix:**
```bash
pip install 'supereval[pdf]'
```

---

### `ImportError: No module named 'anthropic'`

You're using `--backend anthropic` without the optional dependency.

**Fix:**
```bash
pip install 'supereval[anthropic]'
```

---

## AWS credentials

### `NoCredentialsError: Unable to locate credentials`

boto3 can't find AWS credentials.

**Fix:** Configure credentials using one of:

```bash
# Option 1 — environment variables
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_DEFAULT_REGION=us-east-1

# Option 2 — AWS CLI
aws configure

# Option 3 — IAM role (on EC2/ECS/Lambda)
# No action needed — boto3 auto-detects the instance profile
```

---

### `AccessDeniedException: User is not authorized to perform: bedrock:InvokeModel`

Your credentials don't have Bedrock access.

**Fix:** Attach this policy to your IAM user or role, replacing the resource with the specific model ARN(s):

```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeModel"],
  "Resource": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0"
}
```

---

### `ValidationException: The provided model identifier is invalid` or `ResourceNotFoundException`

The Bedrock model ID is wrong or the model isn't enabled in your region.

**Fix:**
1. Run `supereval providers` to see valid model IDs
2. Check that the model is enabled: AWS Console → Bedrock → Model access → enable the model
3. If using a cross-region inference profile (e.g. `us.anthropic.claude-*`), make sure cross-region inference is enabled in your account

---

## Datasets

### `FileNotFoundError: datasets/my-dataset/dataset.json`

supereval can't find the dataset. Either the dataset doesn't exist or you're running from the wrong directory.

**Fix:**
```bash
# Check what datasets exist
supereval dataset list

# Or set the datasets directory explicitly
export SUPEREVAL_DATASETS_DIR=/path/to/your/datasets
supereval dataset list
```

---

### `ValidationError: X validation error(s)` when adding cases

Your JSONL file has cases that don't match the dataset's schema.

**Common causes:**

| Dataset type | Common mistake |
|---|---|
| `qa` | Missing `input.query` or `expected.ground_truth` |
| `classification` | Missing `input.text` or `expected.label` |
| `instruction` | Missing `input.instruction` or `expected.rubric` |

**Fix:** Check the error message — it identifies the line number and field. Typical issues:

```
# Wrong: using 'question' instead of 'query'
{"input": {"question": "What is SQS?"}, "expected": {"ground_truth": "a queue service"}}

# Right:
{"input": {"query": "What is SQS?"}, "expected": {"ground_truth": "a queue service"}}
```

---

### Classification generation fails with `Classification generation requires labels`

**Fix:** The dataset was created without `--labels`. Recreate it:

```bash
supereval dataset create ticket-router \
  --type classification \
  --labels "billing,networking,storage,iam,compute"
```

There's no `update` command — recreate the dataset and re-add your cases.

---

## Eval runs

### All cases fail immediately

**Symptom:** Pass rate 0% on the first run with no obvious error.

**Common causes:**

1. **Model returning unexpected format** — The default prompt for `classification` asks the model to respond with *only* the label name. If your model adds explanation text, the `icontains` check may still pass. If all classification cases fail, the model may not be recognising the label at all. Try running `promptfoo eval` manually with the exported config to see raw responses:
   ```bash
   supereval dataset export my-dataset --output debug.yaml
   promptfoo eval --config debug.yaml --verbose
   ```

2. **Wrong ground truth** — Check that your `ground_truth` values are exact substrings of what the model would normally output. `"256 KB"` will not match a model that responds `"256kb"` (though the check is case-insensitive, so `"256 KB"` and `"256 kb"` are equivalent).

3. **Promptfoo grading model unavailable** — The LLM judge requires an outbound API call. If you're in a restricted network, all `llm-rubric` asserts will fail. Use a custom Bedrock grading provider — see [Custom Promptfoo config](./custom-config.md).

---

### `promptfoo eval failed (exit 1)`

**Fix:** Run the eval manually to see the full error:

```bash
supereval dataset export my-dataset --output debug.yaml
# Add your prompts/providers to debug.yaml manually, then:
promptfoo eval --config debug.yaml
```

Common causes: invalid model ID, API key not set as environment variable, network issue.

---

### Baseline comparison always says "No baseline found"

`baseline.json` doesn't exist yet, or it's in the wrong directory.

**Fix:**
```bash
# Create the baseline
supereval run my-dataset --model <model> --update-baseline

# Verify it was created
ls datasets/my-dataset/baseline.json
```

---

### `baseline.json` merge conflict in git

Two PRs both updated the baseline. Git can't auto-merge JSON.

**Fix:** Accept one baseline and discard the other, then re-run the eval on the losing branch:

```bash
# Keep the main branch baseline
git checkout main -- datasets/my-dataset/baseline.json

# Re-run on your branch to get fresh results
supereval run my-dataset --model <model> --compare-baseline
```

If both branches genuinely improved the model, update the baseline after merging:

```bash
git checkout main
git merge feature-branch
supereval run my-dataset --model <model> --update-baseline
git add datasets/my-dataset/baseline.json
git commit -m "chore: update baseline post-merge"
```

---

## S3 sources

### `ClientError: Access Denied` when using `--from s3://...`

**Fix:** The IAM role/user needs these permissions on the bucket:

```json
{
  "Effect": "Allow",
  "Action": ["s3:ListBucket", "s3:GetObject"],
  "Resource": [
    "arn:aws:s3:::my-bucket",
    "arn:aws:s3:::my-bucket/*"
  ]
}
```

Note: `s3:ListBucket` goes on the bucket ARN (no trailing `/*`). `s3:GetObject` goes on the objects ARN (with `/*`).

---

### `ValueError: No supported files found at s3://...`

The prefix exists but contains no `.txt`, `.md`, or `.pdf` files.

**Fix:** Check the exact key prefix. S3 prefixes are case-sensitive and must match exactly:

```bash
# List what's actually there
aws s3 ls s3://my-bucket/docs/ --recursive
```

---

## History store (DuckDB)

### `duckdb.IOException: Could not open file ... already opened`

Another process has the `supereval.db` file open (e.g. a previous `supereval` process that didn't exit cleanly).

**Fix:**
```bash
# Find and kill the process holding the file
lsof supereval.db

# Or use a different DB path for this run
SUPEREVAL_DB_PATH=/tmp/supereval-run.db supereval run ...
```

---

### History is empty after running evals

**Symptom:** `supereval history list` shows nothing, but runs completed.

**Causes:**
1. You used `--no-record` on those runs
2. `SUPEREVAL_DB_PATH` pointed to a different file than what you're querying now
3. An error silently failed to write the record (check for the yellow warning in run output)

**Fix:**
```bash
# Check which DB file is being used
echo $SUPEREVAL_DB_PATH  # empty means ./supereval.db

# Check if the DB file exists
ls -la supereval.db
```

---

## Agent eval

### `ValueError: Failed to import 'myapp.agents:MyAgent'`

The module path is wrong or the class doesn't exist.

**Fix:** Test the import manually:
```bash
python3 -c "from myapp.agents import MyAgent; print(MyAgent)"
```

Common mistakes:
- Relative path instead of Python module path: use `myapp.agents:MyAgent`, not `./myapp/agents.py:MyAgent`
- Module not in `PYTHONPATH`: run from the directory where `myapp/` lives, or `pip install -e .` your package

---

### `TypeError: run() missing required argument`

Your `AgentRunner` class has `__init__` parameters with no defaults.

**Fix:** Either add default values or create a zero-argument wrapper:

```python
class MyAgentForEval:
    def __init__(self):
        self._agent = MyAgent(model="claude-opus-4-6", region="us-east-1")

    def run(self, task, tool_executor):
        return self._agent.run(task, tool_executor)
```

---

### All agent cases fail with `Agent error: ...`

Your `run()` method is raising an exception. supereval catches it and records the exception message in `trajectory.error`.

**Fix:** Add exception handling inside your runner, or reproduce the crash manually:

```python
from supereval.agent.runner import MockToolExecutor
from supereval.agent.models import AgentTestCase, AgentTask, MockResponse
import json

# Load a failing test case
case = AgentTestCase(
    input=AgentTask(task="Which region is my-bucket in?"),
    tools={"search": MockResponse(response=[{"bucket": "my-bucket", "region": "us-west-2"}])},
    expected={},
)
executor = MockToolExecutor(case.tools, available_tools=[])
agent = MyAgent()
trajectory = agent.run(case.input.task, executor)
print(trajectory)
```
