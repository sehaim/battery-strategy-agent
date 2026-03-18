import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import operator
from state import AgentState


def test_state_keys():
    required = [
        "query", "lg_rag_result", "catl_rag_result",
        "web_positive", "web_negative", "web_neutral", "market_background",
        "merged_data", "swot_lg", "swot_catl",
        "comparison", "implications", "draft_report", "final_report", "references",
        "quality_passed", "revision_feedback", "revision_count",
        "error_log", "retry_count", "max_retries",
        "current_step", "fallback_triggered", "is_complete",
    ]
    hints = AgentState.__annotations__
    for k in required:
        assert k in hints, f"State 필드 누락: {k}"
    print(f"✅ State 필드 검증 통과 ({len(required)}개)")


def test_error_log_accumulate():
    """error_log는 Annotated[List, operator.add] — 누적 동작 확인"""
    a = ["에러1"]
    b = ["에러2"]
    result = operator.add(a, b)
    assert result == ["에러1", "에러2"]
    print("✅ error_log 누적 동작 확인")


def test_initial_state():
    state: AgentState = {
        "query": "테스트",
        "lg_rag_result": None, "catl_rag_result": None,
        "web_positive": None, "web_negative": None,
        "web_neutral": None, "market_background": None,
        "merged_data": None,
        "swot_lg": None, "swot_catl": None,
        "comparison": None, "implications": None,
        "draft_report": None, "final_report": None, "references": [],
        "quality_passed": False, "revision_feedback": None, "revision_count": 0,
        "error_log": [], "retry_count": 0, "max_retries": 2,
        "current_step": "start", "fallback_triggered": False, "is_complete": False,
    }
    assert state["max_retries"] == 2
    assert state["retry_count"] == 0
    assert state["current_step"] == "start"
    print("✅ 초기 State 생성 확인")


def test_supervisor_routing():
    from agents.supervisor import route_next

    cases = [
        ({"current_step": "start"},           "lg_rag"),
        ({"current_step": "lg_rag_done"},     "catl_rag"),
        ({"current_step": "catl_rag_done"},   "web_search"),
        ({"current_step": "web_search_done"}, "aggregator"),
        ({"current_step": "aggregator_done"}, "swot"),
        ({"current_step": "swot_done"},       "insight"),
        ({"current_step": "insight_done"},    "reflect"),
        ({"current_step": "error"},           "error_handler"),
        ({"current_step": "report_done"},     "done"),
    ]
    for state, expected in cases:
        got = route_next(state)
        assert got == expected, f"라우팅 오류: {state} → {got} (기대: {expected})"
    print(f"✅ Supervisor 라우팅 {len(cases)}개 케이스 통과")


def test_reflect_routing():
    from agents.supervisor import route_next
    pass_state = {"current_step": "reflect_done", "quality_passed": True}
    fail_state = {"current_step": "reflect_done", "quality_passed": False}
    assert route_next(pass_state) == "report_writer"
    assert route_next(fail_state) == "swot"
    print("✅ Reflect PASS/FAIL 라우팅 확인")


def test_error_handler():
    from agents.error_handler import error_handler_node
    # retry 가능
    s1 = {"retry_count": 0, "max_retries": 2, "error_log": ["lg_rag: timeout"], "current_step": "error"}
    r1 = error_handler_node(s1)
    assert r1["retry_count"] == 1
    assert r1["current_step"] == "lg_rag"
    # fallback
    s2 = {"retry_count": 2, "max_retries": 2, "error_log": ["lg_rag: timeout"], "current_step": "error"}
    r2 = error_handler_node(s2)
    assert r2["fallback_triggered"] is True
    print("✅ Error Handler retry / fallback 동작 확인")


if __name__ == "__main__":
    test_state_keys()
    test_error_log_accumulate()
    test_initial_state()
    test_supervisor_routing()
    test_reflect_routing()
    test_error_handler()
    print("\n✅ 전체 테스트 통과")
