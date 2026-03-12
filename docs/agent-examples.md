# Agent Runner Reference Implementations

Three reference implementations are provided in `examples/`. They show how to wrap common frameworks in the `AgentRunner` protocol. Copy and adapt whichever fits your stack.

---

## Choosing a runner

| Runner | File | Use when |
|---|---|---|
| `ReActAgent` | `react_bedrock.py` | You want full control over the agent loop; no framework dependency |
| `LangChainAgentRunner` | `langchain.py` | Your agent is already built on LangChain |
| `BedrockAgentRunner` | `bedrock_agents.py` | You have a deployed Bedrock Agent and want integration tests |

The first two run fully offline against mock tools — they are deterministic, free, and suitable for CI. The third calls real AWS infrastructure.

---

## `ReActAgent` — custom ReAct loop on Bedrock

**File:** `examples/react_bedrock.py`

**What it does:**
Implements a Reason + Act loop from scratch. The model is prompted to output XML-tagged tokens (`<thought>`, `<tool_call>`, `<answer>`). The runner parses each response, executes tool calls through `MockToolExecutor`, and feeds observations back as `<observation>` messages. It loops until the model emits `<answer>` or hits `max_iterations`.

**Usage:**
```bash
supereval agent run my-dataset \
  --runner examples.react_bedrock:ReActAgent
```

**Configuration:**
```python
# Override defaults at construction time
agent = ReActAgent(
    model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
    region="us-east-1",
    max_iterations=10,
)
```

Pass a custom instance by creating a wrapper class:

```python
# myapp/agents.py
from agent_eval.examples.react_bedrock import ReActAgent as _Base

class MyReActAgent(_Base):
    def __init__(self):
        super().__init__(
            model_id="us.anthropic.claude-opus-4-6-20251101-v1:0",
            max_iterations=15,
        )
```

```bash
supereval agent run my-dataset --runner myapp.agents:MyReActAgent
```

**How trajectory steps map to the output:**

| Model output | Step type |
|---|---|
| `<thought>...</thought>` | `StepType.thought` |
| `<tool_call>{"name": ..., "arguments": ...}</tool_call>` | `StepType.tool_call` |
| Tool response | `StepType.observation` |
| `<answer>...</answer>` | `StepType.answer` |

If `max_iterations` is reached without an `<answer>` tag, the trajectory returns with `error` set and no `final_answer`.

**Tradeoffs:**
- Full control over parsing, prompting, and loop logic
- No additional dependencies beyond boto3
- Prompt format is XML-specific — you'd need to adapt if your production agent uses a different format

---

## `LangChainAgentRunner` — LangChain agent with mock tools

**File:** `examples/langchain.py`

**What it does:**
Wraps a LangChain `AgentExecutor` so it evaluates against `MockToolExecutor`. Each tool spec from the dataset is converted into a LangChain `Tool` that delegates to `tool_executor.execute()`. This means your agent calls its normal LangChain tool-use flow but gets back canned responses — runs are deterministic and don't touch real services.

**Installation:**
```bash
pip install langchain langchain-aws
```

**Usage:**
```bash
supereval agent run my-dataset \
  --runner examples.langchain:LangChainAgentRunner
```

**Swapping the LLM:**
The runner uses `ChatBedrock` by default. To use a different provider, copy the file and change the `llm` initialisation:

```python
# Use Anthropic API directly
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-opus-4-6")

# Use OpenAI
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o")
```

**How trajectory steps are recorded:**
Tool calls and observations are recorded as they happen inside the LangChain callback. The `final_answer` comes from `executor.invoke()["output"]`. Internal LangChain thought steps are not currently recorded as `StepType.thought` — only tool calls, observations, and the final answer appear in the trajectory.

**Tradeoffs:**
- Requires LangChain dependencies
- Agent logic matches your production code exactly (no reimplementation)
- Thought steps are not fully captured (LangChain abstracts them)
- Tool input parsing assumes JSON; falls back to `{"input": raw_string}` on parse failure

---

## `BedrockAgentRunner` — AWS Bedrock Agents

**File:** `examples/bedrock_agents.py`

**What it does:**
Invokes a deployed Bedrock Agent via `bedrock-agent-runtime:InvokeAgent` and converts its trace events into a `Trajectory`. The agent runs against its real Lambda action groups — mock tools are bypassed entirely.

**Usage:**
```bash
export BEDROCK_AGENT_ID=XXXXXXXXXX
export BEDROCK_AGENT_ALIAS_ID=YYYYYYYYYY

supereval agent run my-dataset \
  --runner examples.bedrock_agents:BedrockAgentRunner
```

Or pass IDs directly:
```python
class MyBedrockRunner(BedrockAgentRunner):
    def __init__(self):
        super().__init__(
            agent_id="XXXXXXXXXX",
            agent_alias_id="YYYYYYYYYY",
            region="us-west-2",
        )
```

**Important:** Because this runner calls real AWS infrastructure:
- The `tools` block in each `AgentTestCase` is **ignored** — the agent uses its real Lambda functions
- Cost and latency figures are real AWS charges
- Runs are not deterministic — the agent may behave differently on each invocation
- Do not use this runner in CI unless you have a dedicated staging environment

**Trace events mapped to steps:**

| Bedrock trace event | Step type |
|---|---|
| `orchestrationTrace.rationale.text` | `StepType.thought` |
| `orchestrationTrace.invocationInput.actionGroupInvocationInput` | `StepType.tool_call` |
| `orchestrationTrace.observation.actionGroupInvocationOutput` | `StepType.observation` |
| Final completion chunk | `StepType.answer` |

**When to use this vs. `ReActAgent`:**

Use `BedrockAgentRunner` when you want to verify that your deployed Bedrock Agent behaves correctly end-to-end in a staging environment. Use `ReActAgent` (or `LangChainAgentRunner`) for fast, reproducible, offline evals in CI.

**IAM requirements:**
```json
{
  "Effect": "Allow",
  "Action": ["bedrock:InvokeAgent"],
  "Resource": "arn:aws:bedrock:*:*:agent-alias/*/*"
}
```

---

## Implementing your own runner

If none of the reference implementations match your stack, implement the `AgentRunner` protocol directly:

```python
from supereval.agent.models import Step, StepType, ToolCall, Trajectory

class MyAgent:
    def run(self, task: str, tool_executor) -> Trajectory:
        steps = []

        # Call your agent logic here
        # Use tool_executor.execute(tool_name, arguments) to call tools
        # Use tool_executor.available_tools() to get ToolSpec list

        result = tool_executor.execute("search", {"query": task})
        steps.append(Step(
            type=StepType.tool_call,
            content="searched",
            tool_call=ToolCall(name="search", arguments={"query": task}, result=result),
        ))
        steps.append(Step(type=StepType.observation, content=str(result)))

        answer = str(result[0]) if result else "not found"
        steps.append(Step(type=StepType.answer, content=answer))

        return Trajectory(steps=steps, final_answer=answer)
```

The only requirement is that `run()` returns a `Trajectory`. If your agent crashes, catch the exception and return `Trajectory(error=str(exc))` — the scorer will mark the case as failed with the error message as the failure reason.
