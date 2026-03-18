import os
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import REFLECT_PROMPT

logger = logging.getLogger(__name__)


def reflect_node(state: AgentState) -> dict:
    logger.info("[Reflect Agent] 품질 검토 시작")
    try:
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
        draft = state.get("draft_report") or state.get("final_report") or ""

        # Reflect Balance Audit — 긍정/비판 비율 간단 체크
        balance_warning = ""
        pos = state.get("web_positive", "") or ""
        neg = state.get("web_negative", "") or ""
        if pos and neg:
            total = len(pos) + len(neg)
            if total > 0 and (len(pos) / total > 0.7 or len(neg) / total > 0.7):
                balance_warning = "\n[경고] 긍정/비판 서술 비율이 70%를 초과했습니다. 균형을 맞추세요."

        response = llm.invoke([
            SystemMessage(content="당신은 보고서 품질 검토 전문가입니다. 반드시 JSON만 출력하세요."),
            HumanMessage(content=REFLECT_PROMPT.format(draft=draft[:6000]) + balance_warning),
        ]).content

        try:
            clean = response.strip()
            if "```" in clean:
                clean = clean.split("```")[1]
                if clean.startswith("json"):
                    clean = clean[4:]
            result = json.loads(clean)
            quality_passed    = bool(result.get("quality_passed", False))
            revision_feedback = result.get("feedback", "")
        except json.JSONDecodeError:
            logger.warning("Reflect JSON 파싱 실패 → FAIL 처리")
            quality_passed    = False
            revision_feedback = f"파싱 실패: {response[:300]}"

        revision_count = state.get("revision_count", 0)

        # revision 2회 초과 시 강제 PASS (fallback)
        if not quality_passed and revision_count >= 2:
            logger.warning("[Reflect] revision 한도 초과 → fallback PASS")
            quality_passed    = True
            revision_feedback = ""

        logger.info(f"[Reflect] 결과: {'PASS' if quality_passed else 'FAIL'}")
        return {
            "quality_passed":    quality_passed,
            "revision_feedback": revision_feedback,
            "revision_count":    revision_count + (0 if quality_passed else 1),
            "current_step":      "reflect_done",
        }
    except Exception as e:
        logger.error(f"[Reflect Agent] 에러: {e}")
        return {
            "error_log":    [f"reflect: {str(e)}"],
            "current_step": "error",
        }
