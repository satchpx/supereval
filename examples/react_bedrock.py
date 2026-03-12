"""
Reference AgentRunner: custom ReAct loop on Amazon Bedrock.

This is a minimal but complete example of how to wrap a ReAct-style agent
(Reason + Act) in the AgentRunner protocol so it can be evaluated by supereval.

Usage
-----
supereval agent run my-dataset --runner examples.react_bedrock:ReActAgent

Prerequisites
-------------
- AWS credentials configured (env vars, ~/.aws/credentials, or IAM role)
- Bedrock model enabled in your account (default: Claude 3.5 Sonnet)

How it works
------------
1. Agent is given the task and the list of available tools.
2. It calls the model in a loop:
   a. Model emits a <thought> and optionally a <tool_call>.
   b. If a tool call is requested, MockToolExecutor.execute() is called.
   c. The result is fed back as an observation.
   d. When the model emits a <answer> tag, the loop ends.
3. The full trajectory is returned to supereval for scoring.

This is intentionally simple — it does not handle:
- Streaming responses
- Parallel tool calls
- Multi-turn memory beyond the current trajectory
"""
from __future__ import annotations

import json
import re

import boto3

from supereval.agent.models import Step, StepType, ToolCall, Trajectory

# Import the protocols (no concrete dependency — just for type hints)
try:
    from supereval.agent.runner import AgentRunner, ToolExecutor
except ImportError:
    AgentRunner = object  # type: ignore
    ToolExecutor = object  # type: ignore

_DEFAULT_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_DEFAULT_REGION = "us-east-1"
_MAX_ITERATIONS = 10

_SYSTEM_PROMPT = """\
You are a helpful assistant that solves tasks step by step using available tools.

For each step, output ONE of:
  <thought>your reasoning here</thought>
  <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
  <answer>your final answer</answer>

Always think before acting. Stop as soon as you have enough information to answer."""


class ReActAgent:
    """
    Minimal ReAct agent on Amazon Bedrock.
    Implements the AgentRunner protocol.
    """

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        region: str = _DEFAULT_REGION,
        max_iterations: int = _MAX_ITERATIONS,
    ):
        self.model_id = model_id
        self.region = region
        self.max_iterations = max_iterations
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _tool_descriptions(self, tool_executor: ToolExecutor) -> str:
        lines = ["Available tools:"]
        for spec in tool_executor.available_tools():
            lines.append(f"  - {spec.name}: {spec.description}")
            if spec.parameters:
                lines.append(f"    parameters: {json.dumps(spec.parameters)}")
        return "\n".join(lines)

    def _call_model(self, messages: list[dict]) -> str:
        client = self._get_client()
        body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": _SYSTEM_PROMPT,
            "messages": messages,
        })
        response = client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        return json.loads(response["body"].read())["content"][0]["text"]

    def run(self, task: str, tool_executor: ToolExecutor) -> Trajectory:
        steps: list[Step] = []
        total_cost = 0.0
        total_latency = 0

        tool_desc = self._tool_descriptions(tool_executor)
        messages = [
            {
                "role": "user",
                "content": f"{tool_desc}\n\nTask: {task}",
            }
        ]

        for _ in range(self.max_iterations):
            import time
            t0 = time.monotonic()
            response_text = self._call_model(messages)
            latency_ms = int((time.monotonic() - t0) * 1000)
            total_latency += latency_ms

            messages.append({"role": "assistant", "content": response_text})

            # Parse the model's output
            thought = re.search(r"<thought>(.*?)</thought>", response_text, re.DOTALL)
            tool_call_match = re.search(
                r"<tool_call>(.*?)</tool_call>", response_text, re.DOTALL
            )
            answer_match = re.search(r"<answer>(.*?)</answer>", response_text, re.DOTALL)

            if thought:
                steps.append(Step(
                    type=StepType.thought,
                    content=thought.group(1).strip(),
                    latency_ms=latency_ms,
                ))

            if tool_call_match:
                try:
                    call_data = json.loads(tool_call_match.group(1).strip())
                    tool_name = call_data.get("name", "")
                    arguments = call_data.get("arguments", {})

                    try:
                        result = tool_executor.execute(tool_name, arguments)
                        tc = ToolCall(name=tool_name, arguments=arguments, result=result)
                        observation = json.dumps(result)
                    except Exception as e:
                        tc = ToolCall(name=tool_name, arguments=arguments, error=str(e))
                        observation = f"Error: {e}"

                    steps.append(Step(
                        type=StepType.tool_call,
                        content=f"Calling {tool_name}",
                        tool_call=tc,
                    ))
                    steps.append(Step(
                        type=StepType.observation,
                        content=observation,
                    ))
                    messages.append({
                        "role": "user",
                        "content": f"<observation>{observation}</observation>",
                    })
                except json.JSONDecodeError as e:
                    steps.append(Step(
                        type=StepType.thought,
                        content=f"[parse error in tool call: {e}]",
                    ))

            if answer_match:
                final_answer = answer_match.group(1).strip()
                steps.append(Step(
                    type=StepType.answer,
                    content=final_answer,
                    latency_ms=latency_ms,
                ))
                return Trajectory(
                    steps=steps,
                    final_answer=final_answer,
                    total_latency_ms=total_latency,
                    total_cost_usd=total_cost,
                )

        # Max iterations reached without an answer
        return Trajectory(
            steps=steps,
            error=f"Reached max iterations ({self.max_iterations}) without producing an answer",
            total_latency_ms=total_latency,
            total_cost_usd=total_cost,
        )
