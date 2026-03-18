import os
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import INSIGHT_PROMPT

logger = logging.getLogger(__name__)


def insight_node(state: AgentState) -> dict:
    logger.info("[Insight Agent] 시작")
    try:
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.2)

        response = llm.invoke([
            SystemMessage(content="당신은 배터리 산업 전략 분석가입니다."),
            HumanMessage(content=INSIGHT_PROMPT.format(
                merged_data=str(state.get("merged_data", ""))[:3000],
                swot_lg=str(state.get("swot_lg", {})),
                swot_catl=str(state.get("swot_catl", {})),
            )),
        ]).content

        # 비교 + 시사점 분리
        lines = response.split("\n")
        mid = len(lines) // 2
        comparison   = "\n".join(lines[:mid])
        implications = "\n".join(lines[mid:])

        logger.info("[Insight Agent] 완료")
        return {
            "comparison":   comparison,
            "implications": implications,
            "current_step": "insight_done",
        }
    except Exception as e:
        logger.error(f"[Insight Agent] 에러: {e}")
        return {
            "error_log":    [f"insight: {str(e)}"],
            "current_step": "error",
        }
