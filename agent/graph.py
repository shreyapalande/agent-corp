from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    load_cache_node,
    news_node,
    funding_node,
    techstack_node,
    competitor_node,
    people_node,
    product_node,
    synthesize_node,
    change_detection_node,
)


def build_graph():
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("load_cache_node", load_cache_node)
    workflow.add_node("news_node", news_node)
    workflow.add_node("funding_node", funding_node)
    workflow.add_node("techstack_node", techstack_node)
    workflow.add_node("competitor_node", competitor_node)
    workflow.add_node("people_node", people_node)
    workflow.add_node("product_node", product_node)
    workflow.add_node("synthesize_node", synthesize_node)
    workflow.add_node("change_detection_node", change_detection_node)

    # Fan-out: START triggers load_cache_node + all 6 search nodes in parallel
    for node in ["load_cache_node", "news_node", "funding_node", "techstack_node", "competitor_node", "people_node", "product_node"]:
        workflow.add_edge(START, node)

    # Fan-in: load_cache_node + all 6 search nodes must all complete before synthesize_node
    # This guarantees the old cache is in state before synthesize_node overwrites it
    for node in ["load_cache_node", "news_node", "funding_node", "techstack_node", "competitor_node", "people_node", "product_node"]:
        workflow.add_edge(node, "synthesize_node")

    # change_detection_node always runs last, after synthesis
    workflow.add_edge("synthesize_node", "change_detection_node")
    workflow.add_edge("change_detection_node", END)

    return workflow.compile()
