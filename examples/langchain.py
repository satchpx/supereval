"""
Reference AgentRunner: LangChain tool-use agent.

Wraps a LangChain agent executor so it evaluates against supereval's
MockToolExecutor, giving you reproducible offline evals without hitting
real APIs or external services.

Usage
-----
supereval agent run my-dataset --runner examples.langchain:LangChainAgentRunner

Prerequisites
-------------
pip install langchain langchain-aws

Set AWS credentials for Bedrock (or swap ChatBedrock for ChatOpenAI / ChatAnthropic).

How it works
------------
Each tool in the AgentTestCase.tools block is wrapped in a LangChain Tool that
delegates to MockToolExecutor.execute(). This means the agent calls its normal
tool-use flow but gets back canned responses, making runs deterministic and free.
"""
from __future__ import annotations

import json

from supereval.agent.models import Step, StepType, ToolCall, Trajectory

try:
    from supereval.agent.runner import AgentRunner, ToolExecutor
except ImportError:
    AgentRunner = object  # type: ignore
    ToolExecutor = object  # type: ignore


class LangChainAgentRunner:
    """
    LangChain agent executor backed by MockToolExecutor.
    Implements the AgentRunner protocol.

    Install dependencies:
        pip install langchain langchain-aws

    Swap the model init below for your preferred LLM provider.
    """

    def __init__(
        self,
        model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0",
        region: str = "us-east-1",
    ):
        self.model_id = model_id
        self.region = region

    def run(self, task: str, tool_executor: ToolExecutor) -> Trajectory:
        try:
            from langchain.agents import AgentExecutor, create_tool_calling_agent
            from langchain.prompts import ChatPromptTemplate
            from langchain.tools import Tool
            from langchain_aws import ChatBedrock
        except ImportError as e:
            raise ImportError(
                "LangChain dependencies not installed. "
                "Run: pip install langchain langchain-aws"
            ) from e

        steps: list[Step] = []

        # Wrap each mock tool as a LangChain Tool
        lc_tools = []
        for spec in tool_executor.available_tools():
            # Capture spec.name in the closure
            def make_tool_fn(tool_name: str):
                def tool_fn(input_str: str) -> str:
                    try:
                        # Try to parse JSON input; fall back to {"input": input_str}
                        try:
                            args = json.loads(input_str)
                        except (json.JSONDecodeError, TypeError):
                            args = {"input": input_str}

                        result = tool_executor.execute(tool_name, args)
                        steps.append(Step(
                            type=StepType.tool_call,
                            content=f"Called {tool_name}",
                            tool_call=ToolCall(
                                name=tool_name,
                                arguments=args,
                                result=result,
                            ),
                        ))
                        steps.append(Step(
                            type=StepType.observation,
                            content=str(result),
                        ))
                        return json.dumps(result)
                    except Exception as e:
                        steps.append(Step(
                            type=StepType.tool_call,
                            content=f"Called {tool_name}",
                            tool_call=ToolCall(
                                name=tool_name,
                                arguments={},
                                error=str(e),
                            ),
                        ))
                        return f"Error: {e}"
                return tool_fn

            lc_tools.append(
                Tool(
                    name=spec.name,
                    description=spec.description,
                    func=make_tool_fn(spec.name),
                )
            )

        llm = ChatBedrock(
            model_id=self.model_id,
            region_name=self.region,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Use tools to answer questions accurately."),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(llm, lc_tools, prompt)
        executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=False)

        try:
            import time
            t0 = time.monotonic()
            agent_response = executor.invoke({"input": task})
            elapsed_ms = int((time.monotonic() - t0) * 1000)

            final_answer = agent_response.get("output", "")
            steps.append(Step(type=StepType.answer, content=final_answer))

            return Trajectory(
                steps=steps,
                final_answer=final_answer,
                total_latency_ms=elapsed_ms,
            )
        except Exception as e:
            return Trajectory(steps=steps, error=str(e))
