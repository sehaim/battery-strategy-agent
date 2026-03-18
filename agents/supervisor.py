import logging
from state import AgentState

logger = logging.getLogger(__name__)

PIPELINE = [
    "lg_rag",
    "catl_rag",
    "web_search",
    "aggregator",
    "swot",
    "insight",
    "reflect",
    "report_writer",
]


def supervisor_node(state: AgentState) -> dict:
    step = state.get("current_step", "start")
    logger.info(f"[Supervisor] current_step={step}")

    # 에러 발생 시 error_handler로 라우팅
    if step == "error":
        return {"current_step": "error_handler"}

    # 완료
    if step in ("report_done",):
        return {"current_step": "done", "is_complete": True}

    # 파이프라인 순서대로 다음 스텝 결정
    next_step = _next(step)
    logger.info(f"[Supervisor] → {next_step}")
    return {"current_step": next_step}


def _next(step: str) -> str:
    """현재 step 기준으로 다음 실행할 노드 반환"""
    done_map = {
        "start":           "lg_rag",
        "lg_rag_done":     "catl_rag",
        "catl_rag_done":   "web_search",
        "web_search_done": "aggregator",
        "aggregator_done": "swot",
        "swot_done":       "insight",
        "insight_done":    "reflect",
        "reflect_done":    _reflect_next,
        "error_handler":   "error_handler",
    }
    val = done_map.get(step, "done")
    # callable이면 호출
    return val if isinstance(val, str) else val()


def _reflect_next():
    # reflect_node가 quality_passed를 직접 state에 쓰므로
    # supervisor는 quality_passed 필드를 route 함수에서 확인
    return "report_writer"


def route_next(state: AgentState) -> str:
    """add_conditional_edges 라우팅 함수"""
    step = state.get("current_step", "start")
    logger.debug(f"[route_next] {step}")

    if step == "done":
        return "done"
    if step == "error_handler":
        return "error_handler"

    # Reflect FAIL → swot 재실행
    if step == "reflect_done":
        if state.get("quality_passed"):
            return "report_writer"
        else:
            return "swot"

    # error_handler에서 결정된 다음 step
    if step in ("lg_rag", "catl_rag", "web_search",
                "aggregator", "swot", "insight",
                "reflect", "report_writer"):
        return step

    # done_map 기반
    mapping = {
        "start":           "lg_rag",
        "lg_rag_done":     "catl_rag",
        "catl_rag_done":   "web_search",
        "web_search_done": "aggregator",
        "aggregator_done": "swot",
        "swot_done":       "insight",
        "insight_done":    "reflect",
        "report_done":     "done",
        "error":           "error_handler",
    }
    return mapping.get(step, "done")
