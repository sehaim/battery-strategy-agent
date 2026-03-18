import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import AGGREGATOR_PROMPT

logger = logging.getLogger(__name__)


def aggregator_node(state: AgentState) -> dict:
    logger.info("[Aggregator] 병합 시작")
    try:
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.1)

        lg_data   = state.get("lg_rag_result")   or "LG RAG 결과 없음"
        catl_data = state.get("catl_rag_result") or "CATL RAG 결과 없음"
        web_data  = state.get("web_neutral")     or state.get("web_positive") or "웹서치 결과 없음"

        prompt = AGGREGATOR_PROMPT.format(
            lg_data=lg_data[:3000],
            catl_data=catl_data[:3000],
            web_data=web_data[:2000],
        )
        merged = llm.invoke([
            SystemMessage(content="당신은 배터리 산업 분석 전문가입니다."),
            HumanMessage(content=prompt),
        ]).content

        logger.info("[Aggregator] 완료")
        return {"merged_data": merged, "current_step": "aggregator_done"}
    except Exception as e:
        logger.error(f"[Aggregator] 에러: {e}")
        return {
            "error_log":    [f"aggregator: {str(e)}"],
            "current_step": "error",
        }
