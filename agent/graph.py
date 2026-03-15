from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    news_node,
    funding_node,
    techstack_node,
    competitor_node,
    people_node,
    synthesize_node,
)


def build_graph():
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("news_node", news_node)
    workflow.add_node("funding_node", funding_node)
    workflow.add_node("techstack_node", techstack_node)
    workflow.add_node("competitor_node", competitor_node)
    workflow.add_node("people_node", people_node)
    workflow.add_node("synthesize_node", synthesize_node)

    # Fan-out: START triggers all 5 search nodes in parallel
    for node in ["news_node", "funding_node", "techstack_node", "competitor_node", "people_node"]:
        workflow.add_edge(START, node)

    # Fan-in: all 5 search nodes feed into synthesize_node
    # LangGraph waits for ALL predecessors before executing synthesize_node
    for node in ["news_node", "funding_node", "techstack_node", "competitor_node", "people_node"]:
        workflow.add_edge(node, "synthesize_node")

    workflow.add_edge("synthesize_node", END)

    return workflow.compile()
