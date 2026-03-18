import logging
from state import AgentState
from agents.rag_base import build_store, run_rag
from prompts.prompts import LG_RAG_PROMPT

logger = logging.getLogger(__name__)

DATA_DIR   = "data/lg"
CACHE_PATH = ".cache/lg_index"

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = build_store(DATA_DIR, CACHE_PATH)
    return _store


def lg_rag_node(state: AgentState) -> dict:
    logger.info("[LG RAG] 시작")
    try:
        store = _get_store()
        result = run_rag(
            store=store,
            query=state["query"],
            prompt_template=LG_RAG_PROMPT,
            fallback_msg="data/lg/ 에 PDF 없음. 웹서치 결과로 대체 예정.",
        )
        logger.info("[LG RAG] 완료")
        return {"lg_rag_result": result, "current_step": "lg_rag_done"}
    except Exception as e:
        logger.error(f"[LG RAG] 에러: {e}")
        return {
            "error_log": [f"lg_rag: {str(e)}"],
            "current_step": "error",
        }
