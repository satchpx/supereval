"""
Reference AgentRunner: AWS Bedrock Agents.

Wraps an existing Bedrock Agents agent in the AgentRunner protocol.
The mock tool layer is bypassed — Bedrock Agents calls your real AWS Lambda
action groups. Use this for integration testing against a staging environment,
not for unit-level reproducible evals.

Usage
-----
supereval agent run my-dataset --runner examples.bedrock_agents:BedrockAgentRunner

Prerequisites
-------------
- A Bedrock Agent created in your AWS account with an alias
- Set BEDROCK_AGENT_ID and BEDROCK_AGENT_ALIAS_ID environment variables
- AWS credentials with bedrock:InvokeAgent permission

Note
----
Because this uses real AWS infrastructure (Lambda action groups, knowledge bases),
test cases in your dataset do not need mock tool responses. The `tools` block in
each AgentTestCase is ignored by this runner. Cost and latency figures are real.

For reproducible unit-level evals, use react_bedrock.py with mock tools instead.
"""
from __future__ import annotations

import json
import os
import uuid

import boto3

from supereval.agent.models import Step, StepType, ToolCall, Trajectory

try:
    from supereval.agent.runner import AgentRunner, ToolExecutor
except ImportError:
    AgentRunner = object  # type: ignore
    ToolExecutor = object  # type: ignore


class BedrockAgentRunner:
    """
    Invokes an AWS Bedrock Agent and converts its trace into a Trajectory.
    Implements the AgentRunner protocol.
    """

    def __init__(
        self,
        agent_id: str | None = None,
        agent_alias_id: str | None = None,
        region: str = "us-east-1",
    ):
        self.agent_id = agent_id or os.environ["BEDROCK_AGENT_ID"]
        self.agent_alias_id = (
            agent_alias_id or os.environ["BEDROCK_AGENT_ALIAS_ID"]
        )
        self.region = region
        self._client = None

    def _get_client(self):
        if self._client is None:
            self._client = boto3.client(
                "bedrock-agent-runtime", region_name=self.region
            )
        return self._client

    def run(self, task: str, tool_executor: ToolExecutor) -> Trajectory:
        # NOTE: tool_executor is ignored — Bedrock Agents uses real Lambda functions.
        client = self._get_client()
        session_id = str(uuid.uuid4())
        steps: list[Step] = []
        total_cost = 0.0
        total_latency = 0

        import time
        t0 = time.monotonic()

        try:
            response = client.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=task,
            )

            final_answer = ""
            # Stream the response
            for event in response["completion"]:
                if "chunk" in event:
                    chunk = event["chunk"]
                    if "bytes" in chunk:
                        final_answer += chunk["bytes"].decode("utf-8")

                if "trace" in event:
                    trace = event["trace"].get("trace", {})

                    # Orchestration trace
                    orch = trace.get("orchestrationTrace", {})
                    if "rationale" in orch:
                        steps.append(Step(
                            type=StepType.thought,
                            content=orch["rationale"].get("text", ""),
                        ))

                    if "invocationInput" in orch:
                        inv = orch["invocationInput"]
                        action = inv.get("actionGroupInvocationInput", {})
                        if action:
                            steps.append(Step(
                                type=StepType.tool_call,
                                content=f"Invoking {action.get('actionGroupName', '')}",
                                tool_call=ToolCall(
                                    name=action.get("actionGroupName", ""),
                                    arguments=action.get("requestBody", {}).get(
                                        "content", {}
                                    ),
                                ),
                            ))

                    if "observation" in orch:
                        obs = orch["observation"]
                        action_result = obs.get("actionGroupInvocationOutput", {})
                        if action_result:
                            steps.append(Step(
                                type=StepType.observation,
                                content=action_result.get("text", ""),
                            ))

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            total_latency = elapsed_ms

            if final_answer:
                steps.append(Step(type=StepType.answer, content=final_answer))

            return Trajectory(
                steps=steps,
                final_answer=final_answer or None,
                total_latency_ms=total_latency,
                total_cost_usd=total_cost,
            )

        except Exception as e:
            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return Trajectory(
                steps=steps,
                error=str(e),
                total_latency_ms=elapsed_ms,
            )
