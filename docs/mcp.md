# Using supereval with AI Coding Assistants (MCP)

supereval ships a built-in MCP (Model Context Protocol) server. Once registered, AI coding assistants like Claude Code and Kiro can manage your eval datasets, run evals, generate test cases, and check history — all from a natural language conversation, without you needing to remember any CLI commands.

## What MCP means in practice

You open Claude Code or Kiro and just describe what you want:

> *"Create a QA eval dataset for my AWS support chatbot, generate 20 test cases from docs/aws-kb/, and run them against claude-haiku"*

The assistant calls `create_dataset`, then `generate_cases`, then `run_eval` — automatically, in sequence, with the right arguments — and shows you the results.

Non-technical team members can manage and run evals without learning the CLI. Technical users get a conversational shortcut for multi-step workflows.

---

## Prerequisites

You need:

1. **supereval installed** with the MCP extra:
   ```bash
   pip install 'supereval[mcp]'
   ```
   Verify:
   ```bash
   supereval --version
   supereval mcp --help
   ```

2. **The `supereval` command accessible on your PATH.** This matters because the AI tool spawns `supereval mcp` as a subprocess. Run:
   ```bash
   which supereval   # macOS / Linux
   where supereval   # Windows
   ```
   Note the full path — you'll need it if `supereval` is inside a virtual environment.

3. **Your credentials configured** — see [Environment variables](#environment-variables) below.

---

## Setup: Claude Code

Claude Code reads MCP server configuration from `~/.claude.json` (a file in your home directory, shared across all projects).

### Step 1 — Find or create `~/.claude.json`

| OS | Path |
|---|---|
| macOS / Linux | `~/.claude.json` |
| Windows | `%USERPROFILE%\.claude.json` |

If the file doesn't exist yet, create it. If it already exists (from other MCP servers), add the `supereval` entry inside the existing `"mcpServers"` object.

### Step 2 — Add the supereval server

```json
{
  "mcpServers": {
    "supereval": {
      "command": "supereval",
      "args": ["mcp"]
    }
  }
}
```

If `supereval` is inside a virtual environment and not on your system PATH, use the full path instead:

```json
{
  "mcpServers": {
    "supereval": {
      "command": "/path/to/venv/bin/supereval",
      "args": ["mcp"]
    }
  }
}
```

To find the full path: `which supereval` (macOS/Linux) or `where supereval` (Windows).

### Step 3 — Pass environment variables (if needed)

If you need to pass credentials or set `SUPEREVAL_DATASETS_DIR`, add an `"env"` block:

```json
{
  "mcpServers": {
    "supereval": {
      "command": "supereval",
      "args": ["mcp"],
      "env": {
        "SUPEREVAL_DATASETS_DIR": "/path/to/your/datasets",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_ACCESS_KEY_ID": "...",
        "AWS_SECRET_ACCESS_KEY": "...",
        "ANTHROPIC_API_KEY": "..."
      }
    }
  }
}
```

Only set the keys you actually need. If your credentials come from `~/.aws/credentials`, an IAM role, or system environment variables, you don't need to list them here.

### Step 4 — Restart Claude Code and verify

Restart Claude Code (quit and reopen, not just a new window). Then type:

```
/mcp
```

Claude Code will list all connected MCP servers. You should see:

```
supereval  ●  connected
  Tools: list_datasets, show_dataset, create_dataset, add_cases ...
```

If it shows `● disconnected` or the server isn't listed, see [Troubleshooting](#troubleshooting).

---

## Setup: Kiro

Kiro supports MCP servers both via its UI and via a JSON config file. The UI approach is recommended for first-time setup.

### Option A — Kiro UI (recommended)

1. Open Kiro and click the **MCP** icon in the left sidebar (or open **Settings → MCP Servers**)
2. Click **Add MCP Server**
3. Fill in:
   - **Name:** `supereval`
   - **Command:** `supereval` (or the full path from `which supereval`)
   - **Arguments:** `mcp`
4. Add any environment variables you need (same as the `env` block above)
5. Click **Save** and then **Connect**

The server status indicator will turn green when connected.

### Option B — JSON config file

Kiro also reads from `.kiro/settings/mcp.json` in your project directory (project-scoped) or from the global Kiro settings directory.

**Project-scoped** (`.kiro/settings/mcp.json` in your repo root):
```json
{
  "mcpServers": {
    "supereval": {
      "command": "supereval",
      "args": ["mcp"],
      "env": {
        "SUPEREVAL_DATASETS_DIR": "./datasets"
      }
    }
  }
}
```

After saving, Kiro detects the file automatically — no restart required.

### Verify in Kiro

Open the MCP panel in the sidebar. `supereval` should appear with a green connected indicator and the list of available tools visible under it.

---

## Setup: other MCP-compatible tools

Any tool that supports the MCP stdio transport can use supereval. The configuration format varies by tool, but the server command is always:

```
supereval mcp
```

The server speaks JSON-RPC over stdin/stdout (MCP stdio transport). Refer to your tool's MCP documentation for where to register it.

---

## Environment variables

The MCP server inherits environment variables from the process that starts it. If your credentials and paths are already set in your shell profile, they'll be available automatically.

| Variable | Required? | Purpose |
|---|---|---|
| `SUPEREVAL_DATASETS_DIR` | Recommended | Where datasets live. Defaults to `./datasets` relative to where the server starts — set this explicitly so it always points to the right place. |
| `SUPEREVAL_DB_PATH` | Optional | Path to the DuckDB run history file. Defaults to `./supereval.db`. |
| `AWS_DEFAULT_REGION` | For Bedrock | AWS region for model calls and case generation. |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | For Bedrock | AWS credentials. Not needed if using IAM roles, SSO, or `~/.aws/credentials`. |
| `AWS_PROFILE` | For Bedrock | Named AWS profile to use from `~/.aws/credentials`. |
| `ANTHROPIC_API_KEY` | For Anthropic backend | Required when generating cases with `--backend anthropic`. |
| `OPENAI_API_KEY` | For OpenAI backend | Required when generating cases with `--backend openai`. |

### Setting a permanent `SUPEREVAL_DATASETS_DIR`

Since the MCP server may start from any working directory (depending on which project is open in the editor), it's safest to set an absolute path:

**In `~/.claude.json`:**
```json
"env": {
  "SUPEREVAL_DATASETS_DIR": "/Users/you/projects/myapp/datasets"
}
```

**Or in your shell profile** (`~/.zshrc`, `~/.bashrc`):
```bash
export SUPEREVAL_DATASETS_DIR=/Users/you/projects/myapp/datasets
```

---

## What you can ask

Once connected, describe what you want in natural language. The AI assistant maps your intent to the right tool calls. Some examples:

**Dataset management:**
- *"What eval datasets do I have?"*
- *"Show me the full details and thresholds for aws-support-qa"*
- *"Create a classification dataset called ticket-router with labels: billing, networking, storage, iam"*
- *"Add cases from /tmp/new-cases.jsonl to the aws-support-qa dataset"*

**Running evals:**
- *"Run the aws-support-qa dataset against claude-haiku and tell me the pass rate"*
- *"Run all my datasets against claude-sonnet and compare against baseline"*
- *"Run the qa dataset against anthropic:claude-haiku-4-5-20251001 and update the baseline if it passes"*

**Generating cases:**
- *"Generate 30 test cases from docs/aws-kb/ and stage them for review"*
- *"Generate 20 cases from s3://my-bucket/docs/ using the anthropic backend and auto-import them"*

**History and stats:**
- *"Show me the last 10 eval runs"*
- *"Show the history for aws-support-qa and tell me if pass rates are trending down"*
- *"What's the average cost per run for the qa dataset over the last 5 runs?"*

**Versioning:**
- *"Tag the current aws-support-qa dataset as v1.2.0 before I make changes"*
- *"List all versions of aws-support-qa"*
- *"Restore aws-support-qa to v1.0.0"*

---

## Troubleshooting

### Server shows as disconnected in Claude Code

**Check 1 — Is `supereval mcp` runnable?**
```bash
supereval mcp --help
```
If this errors, `supereval` is not installed or not on PATH.

**Check 2 — Does the JSON parse correctly?**
Invalid JSON in `~/.claude.json` will silently prevent any MCP server from loading. Validate it:
```bash
python3 -m json.tool ~/.claude.json
```

**Check 3 — Is `mcp` installed?**
```bash
pip install 'supereval[mcp]'
```

**Check 4 — Full path required?**
If `supereval` is in a virtual environment, Claude Code may not see it. Use the absolute path:
```bash
which supereval   # copy this output
```
Then in `~/.claude.json`:
```json
"command": "/full/path/to/supereval"
```

---

### `Error: SUPEREVAL_DATASETS_DIR not found` or wrong datasets

The server started from a directory where `./datasets` doesn't exist. Fix: set `SUPEREVAL_DATASETS_DIR` explicitly in the `env` block with an absolute path.

---

### `NoCredentialsError` or Bedrock access errors

The MCP server subprocess doesn't inherit credentials from your terminal session. Pass them explicitly:

```json
"env": {
  "AWS_DEFAULT_REGION": "us-east-1",
  "AWS_ACCESS_KEY_ID": "...",
  "AWS_SECRET_ACCESS_KEY": "..."
}
```

Or if using a named profile:
```json
"env": {
  "AWS_PROFILE": "my-profile",
  "AWS_DEFAULT_REGION": "us-east-1"
}
```

---

### `run_eval` fails with `promptfoo: command not found`

`supereval run` shells out to `promptfoo`. Install it:
```bash
npm install -g promptfoo
```

If npm isn't available, `npx promptfoo@latest` is used automatically as a fallback.

---

### Tools appear in the server but calls return errors

Run the equivalent CLI command directly to isolate the problem:
```bash
supereval dataset list
supereval run my-dataset --model anthropic:claude-haiku-4-5-20251001
```

Errors from the CLI will be more verbose than what the AI tool surfaces. Fix the underlying issue there first.

---

### Claude Code / Kiro keeps asking for confirmation on every tool call

Both tools have configurable trust levels for MCP servers. In Claude Code you can mark a server as trusted so routine read operations (like `list_datasets`, `show_dataset`, `list_history`) don't require confirmation each time. Mutating operations (`run_eval`, `version_restore`, `add_cases`) will still prompt by default, which is intentional.

---

## Security notes

- The MCP server runs with the same permissions as your user account. It can read and write files under `SUPEREVAL_DATASETS_DIR` and make API calls using your credentials.
- Credentials in the `"env"` block of `~/.claude.json` are stored in plaintext. For production or shared machines, use AWS IAM roles, AWS SSO, or a secrets manager instead of embedding keys directly.
- The server only accepts connections from the local MCP client — it does not expose a network port.
