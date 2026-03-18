import logging
from state import AgentState

logger = logging.getLogger(__name__)


def error_handler_node(state: AgentState) -> dict:
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    error_log   = state.get("error_log", [])
    last_error  = error_log[-1] if error_log else "알 수 없는 에러"

    # 에러 유형 분류
    if "timeout" in last_error.lower():
        error_type = "timeout"
    elif "retrieval" in last_error.lower() or "pdf" in last_error.lower():
        error_type = "empty_retrieval"
    elif "json" in last_error.lower() or "parse" in last_error.lower():
        error_type = "parsing_error"
    else:
        error_type = "unknown"

    logger.warning(f"[Error Handler] 에러 유형: {error_type} | 횟수: {retry_count+1}/{max_retries}")

    if retry_count < max_retries:
        # retry: 이전 step으로 복귀
        prev_step = _get_retry_step(state)
        logger.info(f"[Error Handler] retry → {prev_step}")
        return {
            "retry_count":  retry_count + 1,
            "current_step": prev_step,
        }
    else:
        # fallback: 가용 데이터로 보고서 생성
        logger.warning("[Error Handler] 최대 재시도 초과 → fallback")
        return {
            "fallback_triggered": True,
            "current_step":       "report_writer",
        }


def _get_retry_step(state: AgentState) -> str:
    """마지막으로 실패한 단계 추정"""
    error_log = state.get("error_log", [])
    if not error_log:
        return "supervisor"

    last = error_log[-1].lower()
    if "lg_rag" in last:
        return "lg_rag"
    elif "catl_rag" in last:
        return "catl_rag"
    elif "web_search" in last:
        return "web_search"
    elif "aggregator" in last:
        return "aggregator"
    elif "swot" in last:
        return "swot"
    elif "insight" in last:
        return "insight"
    elif "reflect" in last:
        return "reflect"
    return "supervisor"
