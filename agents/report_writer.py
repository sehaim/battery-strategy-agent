import os
import logging
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import REPORT_PROMPT, SUMMARY_PROMPT

logger = logging.getLogger(__name__)

OUTPUT_DIR = "outputs"


def _build_references(state: AgentState) -> str:
    refs = state.get("references") or []
    if not refs:
        return "# REFERENCE\n실제 인용 자료를 여기에 기재하세요."
    lines = ["# REFERENCE"]
    for r in refs:
        lines.append(f"- {r}")
    return "\n".join(lines)


def _to_pdf(md_path: str, pdf_path: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont

        doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        styles = getSampleStyleSheet()
        story = []

        with open(md_path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith("# "):
                    story.append(Paragraph(line[2:], styles["Heading1"]))
                elif line.startswith("## "):
                    story.append(Paragraph(line[3:], styles["Heading2"]))
                elif line.startswith("### "):
                    story.append(Paragraph(line[4:], styles["Heading3"]))
                elif line:
                    story.append(Paragraph(line, styles["Normal"]))
                else:
                    story.append(Spacer(1, 8))

        doc.build(story)
        logger.info(f"PDF 생성: {pdf_path}")
    except Exception as e:
        logger.warning(f"PDF 변환 실패: {e}")


def report_writer_node(state: AgentState) -> dict:
    logger.info("[Report Writer] 보고서 생성 시작")
    try:
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0.2)

        # 보고서 본문 생성
        report = llm.invoke([
            SystemMessage(content="당신은 배터리 산업 전문 보고서 작성자입니다. 마크다운으로 작성하세요."),
            HumanMessage(content=REPORT_PROMPT.format(
                market_background=str(state.get("market_background", ""))[:1500],
                lg_strategy=str(state.get("lg_rag_result", ""))[:2000],
                catl_strategy=str(state.get("catl_rag_result", ""))[:2000],
                comparison=str(state.get("comparison", ""))[:2000],
                implications=str(state.get("implications", ""))[:1500],
                swot_lg=str(state.get("swot_lg", {})),
                swot_catl=str(state.get("swot_catl", {})),
                feedback=str(state.get("revision_feedback", "")),
            )),
        ]).content

        # SUMMARY 생성 (250단어 이내)
        summary = llm.invoke([
            SystemMessage(content="250단어 이내로 핵심 인사이트를 요약하세요."),
            HumanMessage(content=SUMMARY_PROMPT.format(report=report[:4000])),
        ]).content

        # SUMMARY를 보고서 맨 앞에 삽입
        if "# SUMMARY" in report:
            report = report.replace("# SUMMARY", f"# SUMMARY\n\n{summary}", 1)
        else:
            report = f"# SUMMARY\n\n{summary}\n\n{report}"

        # REFERENCE 추가
        report += "\n\n" + _build_references(state)

        # 파일 저장
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        md_path  = f"{OUTPUT_DIR}/battery_strategy_report.md"
        pdf_path = f"{OUTPUT_DIR}/battery_strategy_report.pdf"

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)
        logger.info(f"MD 저장: {md_path}")

        _to_pdf(md_path, pdf_path)

        return {
            "final_report": report,
            "current_step": "report_done",
            "is_complete":  True,
        }
    except Exception as e:
        logger.error(f"[Report Writer] 에러: {e}")
        return {
            "error_log":    [f"report_writer: {str(e)}"],
            "current_step": "error",
        }
