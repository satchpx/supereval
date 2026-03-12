"""
Generates decision_tree.png for use in README.md.
Run from the repo root: python3 generate_diagram.py
Requires: pip install graphviz  +  graphviz CLI (brew install graphviz)
"""

from graphviz import Digraph
import os

dot = Digraph(
    name="eval_tool_decision_tree",
    format="png",
)

dot.attr(rankdir="TB", size="14,12", dpi="150", bgcolor="white", pad="0.5")
dot.attr("graph", fontname="Helvetica")
dot.attr("node", fontname="Helvetica", fontsize="12")
dot.attr("edge", fontname="Helvetica", fontsize="10", color="#555555")

# Style sets
decision = dict(shape="diamond", style="filled", fillcolor="#D6EAF8", color="#2471A3", fontcolor="#1A1A1A")
tool     = dict(shape="box", style="rounded,filled", fillcolor="#D5F5E3", color="#1E8449", fontcolor="#1A1A1A")
start    = dict(shape="oval", style="filled", fillcolor="#FDEBD0", color="#CA6F1E", fontcolor="#1A1A1A")
ref      = dict(shape="box", style="rounded,filled", fillcolor="#EAF2FF", color="#2E86C1", fontcolor="#1A1A1A")

# Nodes
dot.node("start",      "Start here",                            **start)
dot.node("q_when",     "When do you need\nto evaluate?",        **decision)
dot.node("q_what",     "What are you\nevaluating?",             **decision)
dot.node("q_iface",    "Preferred\ninterface?",                 **decision)
dot.node("q_metrics",  "Need built-in\nmetrics library?\n(toxicity, bias,\nhallucination, etc.)", **decision)

dot.node("langfuse",   "Langfuse\nlangfuse.com",                **tool)
dot.node("promptfoo",  "Promptfoo\npromptfoo.dev",              **tool)
dot.node("deepeval",   "DeepEval\nconfident-ai.com",            **tool)
dot.node("braintrust", "Promptfoo or Braintrust\nbraintrust.dev", **tool)
dot.node("ragas",      "RAGAS\ndocs.ragas.io",                  **tool)
dot.node("agent_eval", "supereval agent\n(custom harness)",     **ref)

# Edges
dot.edge("start",     "q_when")
dot.edge("q_when",    "langfuse",   label="  Post-deployment\n  (production monitoring)")
dot.edge("q_when",    "q_what",     label="  Pre-deployment\n  (offline / CI testing)")
dot.edge("q_what",    "q_iface",    label="  Prompts or\n  model selection")
dot.edge("q_what",    "agent_eval", label="  Agent trajectories\n  or tool calls")
dot.edge("q_what",    "ragas",      label="  RAG pipelines")
dot.edge("q_iface",   "promptfoo",  label="  Config-driven\n  (YAML / CLI)")
dot.edge("q_iface",   "q_metrics",  label="  Code-driven\n  (Python)")
dot.edge("q_metrics", "deepeval",   label="  Yes")
dot.edge("q_metrics", "braintrust", label="  No")

output_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(output_dir, "decision_tree")
dot.render(output_path, cleanup=True)
print(f"Generated: {output_path}.png")
