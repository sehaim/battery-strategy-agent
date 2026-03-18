from langgraph.graph import StateGraph, END

from state import AgentState
from agents.supervisor import supervisor_node, route_next
from agents.lg_rag_agent import lg_rag_node
from agents.catl_rag_agent import catl_rag_node
from agents.web_search_agent import web_search_node
from agents.aggregator import aggregator_node
from agents.swot_agent import swot_node
from agents.insight_agent import insight_node
from agents.reflect_agent import reflect_node
from agents.report_writer import report_writer_node
from agents.error_handler import error_handler_node


def build_graph():
    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("supervisor",    supervisor_node)
    graph.add_node("lg_rag",        lg_rag_node)
    graph.add_node("catl_rag",      catl_rag_node)
    graph.add_node("web_search",    web_search_node)
    graph.add_node("aggregator",    aggregator_node)
    graph.add_node("swot",          swot_node)
    graph.add_node("insight",       insight_node)
    graph.add_node("reflect",       reflect_node)
    graph.add_node("report_writer", report_writer_node)
    graph.add_node("error_handler", error_handler_node)

    # 진입점
    graph.set_entry_point("supervisor")

    # Supervisor → 조건부 라우팅
    graph.add_conditional_edges(
        "supervisor",
        route_next,
        {
            "lg_rag":        "lg_rag",
            "catl_rag":      "catl_rag",
            "web_search":    "web_search",
            "aggregator":    "aggregator",
            "swot":          "swot",
            "insight":       "insight",
            "reflect":       "reflect",
            "report_writer": "report_writer",
            "error_handler": "error_handler",
            "done":          END,
        },
    )

    # 서브에이전트 → Supervisor 복귀
    for node in ["lg_rag", "catl_rag", "web_search",
                 "aggregator", "swot", "insight"]:
        graph.add_edge(node, "supervisor")

    # Reflect → 조건부: PASS → report_writer | FAIL → swot
    graph.add_conditional_edges(
        "reflect",
        lambda s: "report_writer" if s.get("quality_passed") else "swot",
        {
            "report_writer": "report_writer",
            "swot":          "swot",
        },
    )

    # Report Writer → END
    graph.add_edge("report_writer", END)

    # Error Handler → Supervisor 복귀
    graph.add_edge("error_handler", "supervisor")

    return graph.compile()


app = build_graph()
