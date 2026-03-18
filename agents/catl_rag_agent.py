import logging
from state import AgentState
from agents.rag_base import build_store, run_rag
from prompts.prompts import CATL_RAG_PROMPT

logger = logging.getLogger(__name__)

DATA_DIR   = "data/catl"
CACHE_PATH = ".cache/catl_index"

_store = None


def _get_store():
    global _store
    if _store is None:
        _store = build_store(DATA_DIR, CACHE_PATH)
    return _store


def catl_rag_node(state: AgentState) -> dict:
    logger.info("[CATL RAG] 시작")
    try:
        store = _get_store()
        result = run_rag(
            store=store,
            query=state["query"],
            prompt_template=CATL_RAG_PROMPT,
            fallback_msg="data/catl/ 에 PDF 없음. 웹서치 결과로 대체 예정.",
        )
        logger.info("[CATL RAG] 완료")
        return {"catl_rag_result": result, "current_step": "catl_rag_done"}
    except Exception as e:
        logger.error(f"[CATL RAG] 에러: {e}")
        return {
            "error_log": [f"catl_rag: {str(e)}"],
            "current_step": "error",
        }
