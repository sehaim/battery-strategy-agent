import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import SWOT_PROMPT

logger = logging.getLogger(__name__)


def swot_node(state: AgentState) -> dict:
    logger.info("[SWOT Agent] 시작")
    try:
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.1)
        merged = state.get("merged_data") or "병합 데이터 없음"

        response = llm.invoke([
            SystemMessage(content="당신은 전략 분석 전문가입니다. 반드시 JSON만 출력하세요."),
            HumanMessage(content=SWOT_PROMPT.format(merged_data=merged[:5000])),
        ]).content

        # JSON 파싱
        try:
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            swot = json.loads(clean)
            swot_lg   = swot.get("lg", {})
            swot_catl = swot.get("catl", {})
        except json.JSONDecodeError:
            logger.warning("SWOT JSON 파싱 실패 → 텍스트로 저장")
            swot_lg   = {"raw": response}
            swot_catl = {"raw": response}
            return {
                "swot_lg":      swot_lg,
                "swot_catl":    swot_catl,
                "error_log":    [f"swot_json_parse: LLM 응답이 유효한 JSON이 아님 — SWOT 구조 불완전할 수 있음"],
                "current_step": "swot_done",
            }

        logger.info("[SWOT Agent] 완료")
        return {
            "swot_lg":      swot_lg,
            "swot_catl":    swot_catl,
            "current_step": "swot_done",
        }
    except Exception as e:
        logger.error(f"[SWOT Agent] 에러: {e}")
        return {
            "error_log":    [f"swot: {str(e)}"],
            "current_step": "error",
        }
