from __future__ import annotations
import os
import re
import logging
from datetime import date
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from state import AgentState
from prompts.prompts import REPORT_PROMPT, SUMMARY_PROMPT, REFERENCE_FORMAT_PROMPT

logger = logging.getLogger(__name__)
OUTPUT_DIR = "outputs"


# ── 문자열 정제 (null byte·제어문자 제거) ─────────────────────
def _clean(text: str, limit: int = 0) -> str:
    """OpenAI API 전송 전 문자열에서 제어 문자와 null byte 제거."""
    # null byte 및 C0/C1 제어 문자 제거 (탭·줄바꿈 제외)
    cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', str(text))
    return cleaned[:limit] if limit else cleaned


# ── 참조 목록 포맷팅 (LLM 사용) ───────────────────────────────
def _build_references(state: AgentState, llm) -> str:
    refs = state.get("references") or []
    if not refs:
        return ""

    # 중복 제거
    unique_refs = list(dict.fromkeys(refs))
    raw = "\n".join(f"- {r}" for r in unique_refs)

    try:
        formatted = llm.invoke([
            SystemMessage(content="당신은 학술 인용 형식 전문가입니다. 지시한 형식대로만 출력하세요."),
            HumanMessage(content=REFERENCE_FORMAT_PROMPT.format(raw_refs=raw)),
        ]).content.strip()
        return f"# REFERENCE\n\n{formatted}"
    except Exception as e:
        logger.warning(f"참조 포맷팅 실패 → 원시 목록 사용: {e}")
        return "# REFERENCE\n\n" + raw


# ── 한국어 폰트 등록 ───────────────────────────────────────────
def _register_korean_font() -> tuple[str, str]:
    """시스템 한국어 폰트를 찾아 ReportLab에 등록. (일반, 볼드) 이름 반환."""
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        # (regular_path, regular_idx, bold_path, bold_idx)
        # AppleGothic — plain TTF, macOS 기본 한글 폰트, 가장 안정적
        ("/System/Library/Fonts/Supplemental/AppleGothic.ttf", None,
         "/System/Library/Fonts/Supplemental/AppleGothic.ttf", None),
        # Apple SD Gothic Neo — TTC (subfont index 0 또는 3 시도)
        ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 0,
         "/System/Library/Fonts/AppleSDGothicNeo.ttc", 6),
        ("/System/Library/Fonts/AppleSDGothicNeo.ttc", 3,
         "/System/Library/Fonts/AppleSDGothicNeo.ttc", 6),
        # NanumGothic
        ("/Library/Fonts/NanumGothic.ttf", None,
         "/Library/Fonts/NanumGothicBold.ttf", None),
        (os.path.expanduser("~/Library/Fonts/NanumGothic.ttf"), None,
         os.path.expanduser("~/Library/Fonts/NanumGothicBold.ttf"), None),
        # Windows Malgun Gothic
        ("C:/Windows/Fonts/malgun.ttf", None,
         "C:/Windows/Fonts/malgunbd.ttf", None),
    ]

    for reg_p, reg_i, bold_p, bold_i in candidates:
        if not os.path.exists(reg_p):
            continue
        try:
            kw_r = {"subfontIndex": reg_i} if reg_i is not None else {}
            pdfmetrics.registerFont(TTFont("KorFont", reg_p, **kw_r))

            bold_name = "KorFont"
            if bold_p and os.path.exists(bold_p):
                kw_b = {"subfontIndex": bold_i} if bold_i is not None else {}
                pdfmetrics.registerFont(TTFont("KorFont-Bold", bold_p, **kw_b))
                bold_name = "KorFont-Bold"

            logger.info(f"한국어 폰트 등록 성공: {reg_p}")
            return "KorFont", bold_name
        except Exception as e:
            logger.warning(f"폰트 등록 실패 ({reg_p}): {e}")

    logger.warning("한국어 폰트 없음 — Helvetica 사용 (한글 깨짐 가능)")
    return "Helvetica", "Helvetica-Bold"


# ── 인라인 마크다운 → ReportLab XML ───────────────────────────
def _md_inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.+?)\*',     r'<i>\1</i>', text)
    text = re.sub(r'`(.+?)`',       r'<font face="Courier">\1</font>', text)
    return text


# ── 마크다운 테이블 → 2D list ──────────────────────────────────
def _parse_table(table_lines: list[str]) -> list[list[str]]:
    rows = []
    for line in table_lines:
        line = line.strip()
        if re.match(r'^\|[\s\-\|:]+\|$', line):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        rows.append(cells)
    return rows


# ── 페이지 번호/헤더 캔버스 ────────────────────────────────────
def _make_canvas_factory(fn: str, fn_bold: str):
    from reportlab.pdfgen import canvas as rcanvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.lib.colors import HexColor

    class NumberedCanvas(rcanvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_states: list[dict] = []

        def showPage(self):
            self._saved_states.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            total = len(self._saved_states)
            for i, state in enumerate(self._saved_states, 1):
                self.__dict__.update(state)
                self._draw_chrome(i, total)
                rcanvas.Canvas.showPage(self)
            rcanvas.Canvas.save(self)

        def _draw_chrome(self, page_num: int, total: int):
            if page_num == 1:
                return  # 표지에는 헤더·푸터 없음
            W, H = A4
            C_NAVY = HexColor("#1a3866")
            C_GRAY = HexColor("#888888")
            # Header line + title
            self.setStrokeColor(C_NAVY)
            self.setLineWidth(1.5)
            self.line(25*mm, H - 14*mm, W - 20*mm, H - 14*mm)
            self.setFont(fn, 8)
            self.setFillColor(C_NAVY)
            self.drawString(25*mm, H - 12*mm,
                            "배터리 산업 전략 분석 보고서 | LG에너지솔루션 vs CATL")
            # Footer
            self.setFont(fn, 8)
            self.setFillColor(C_GRAY)
            self.drawCentredString(W / 2, 12*mm, f"- {page_num} / {total} -")

    return NumberedCanvas


# ── PDF 생성 ───────────────────────────────────────────────────
def _to_pdf(md_path: str, pdf_path: str):
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.lib.colors import HexColor, white
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer,
            Table, TableStyle, HRFlowable, PageBreak,
        )

        fn, fn_bold = _register_korean_font()

        C_NAVY  = HexColor("#1a3866")
        C_BLUE  = HexColor("#3385c7")
        C_LBLUE = HexColor("#ddeeff")
        C_LGRAY = HexColor("#f5f5f5")
        C_MGRAY = HexColor("#cccccc")
        C_TEXT  = HexColor("#222222")
        C_GRAY2 = HexColor("#555555")

        def S(name, **kw) -> ParagraphStyle:
            return ParagraphStyle(name, **kw)

        s_h1     = S("H1",   fontName=fn_bold, fontSize=18, textColor=white,
                      leading=26, spaceAfter=4, spaceBefore=10)
        s_h2     = S("H2",   fontName=fn_bold, fontSize=13, textColor=C_NAVY,
                      leading=19, spaceAfter=3, spaceBefore=12)
        s_h3     = S("H3",   fontName=fn_bold, fontSize=11, textColor=C_BLUE,
                      leading=16, spaceAfter=3, spaceBefore=8)
        s_body   = S("Body", fontName=fn,      fontSize=10, textColor=C_TEXT,
                      leading=16, spaceAfter=5, spaceBefore=1)
        s_bullet = S("Blt",  fontName=fn,      fontSize=10, textColor=C_TEXT,
                      leading=15, spaceAfter=3, leftIndent=14)
        s_ref    = S("Ref",  fontName=fn,      fontSize=8,  textColor=C_GRAY2,
                      leading=12, spaceAfter=2, leftIndent=12)
        s_cover1 = S("Cv1",  fontName=fn_bold, fontSize=30, textColor=C_NAVY,
                      leading=40, alignment=TA_CENTER)
        s_cover2 = S("Cv2",  fontName=fn,      fontSize=16, textColor=C_BLUE,
                      leading=24, alignment=TA_CENTER)
        s_cover3 = S("Cv3",  fontName=fn,      fontSize=11, textColor=C_GRAY2,
                      leading=16, alignment=TA_CENTER)
        s_th     = S("TH",   fontName=fn_bold, fontSize=9,  textColor=white,
                      leading=13, alignment=TA_CENTER)
        s_td     = S("TD",   fontName=fn,      fontSize=9,  textColor=C_TEXT,
                      leading=13)

        W, H = A4
        content_w = W - 45 * mm

        doc = SimpleDocTemplate(
            pdf_path, pagesize=A4,
            leftMargin=25*mm, rightMargin=20*mm,
            topMargin=22*mm, bottomMargin=22*mm,
            title="배터리 산업 전략 분석 보고서",
        )

        story = []

        # ── 표지 (내용 수직 중앙 배치) ──
        # A4 콘텐츠 영역 253mm, 표지 블록 약 64mm → 상단 여백 ≈ (253-64)/2 ≈ 94mm
        story.append(Spacer(1, 94*mm))
        story.append(HRFlowable(width="100%", thickness=2, color=C_NAVY, spaceAfter=8))
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("배터리 산업 전략 분석 보고서", s_cover1))
        story.append(Spacer(1, 6*mm))
        story.append(Paragraph("LG에너지솔루션 vs CATL 비교 분석", s_cover2))
        story.append(Spacer(1, 4*mm))
        story.append(Paragraph(date.today().strftime("%Y년 %m월 %d일"), s_cover3))
        story.append(Spacer(1, 8*mm))
        story.append(HRFlowable(width="100%", thickness=2, color=C_NAVY))
        story.append(PageBreak())

        # ── 본문 파싱 ──
        with open(md_path, encoding="utf-8") as f:
            lines = [l.rstrip("\n") for l in f.readlines()]

        i = 0
        while i < len(lines):
            line = lines[i]

            # H1 (# 으로 시작, ## 아님)
            if re.match(r'^# [^#]', line):
                txt = _md_inline(line[2:].strip())
                cell = Table(
                    [[Paragraph(txt, s_h1)]],
                    colWidths=[content_w],
                )
                cell.setStyle(TableStyle([
                    ("BACKGROUND",    (0,0), (-1,-1), C_NAVY),
                    ("TOPPADDING",    (0,0), (-1,-1), 10),
                    ("BOTTOMPADDING", (0,0), (-1,-1), 10),
                    ("LEFTPADDING",   (0,0), (-1,-1), 14),
                    ("RIGHTPADDING",  (0,0), (-1,-1), 14),
                ]))
                story.append(Spacer(1, 4*mm))
                story.append(cell)
                i += 1
                continue

            # H2
            if re.match(r'^## [^#]', line):
                txt = _md_inline(line[3:].strip())
                story.append(Spacer(1, 3*mm))
                story.append(Paragraph(txt, s_h2))
                story.append(HRFlowable(
                    width="100%", thickness=1.5,
                    color=C_LBLUE, spaceAfter=2,
                ))
                i += 1
                continue

            # H3
            if re.match(r'^### ', line):
                story.append(Paragraph(_md_inline(line[4:].strip()), s_h3))
                i += 1
                continue

            # Markdown table
            if line.startswith("|"):
                tbl_lines = []
                while i < len(lines) and lines[i].startswith("|"):
                    tbl_lines.append(lines[i])
                    i += 1
                rows = _parse_table(tbl_lines)
                if rows:
                    n_cols = max(len(r) for r in rows)
                    col_w  = content_w / n_cols
                    for r in rows:
                        while len(r) < n_cols:
                            r.append("")
                    pdf_rows = []
                    for ri, row in enumerate(rows):
                        style_ = s_th if ri == 0 else s_td
                        pdf_rows.append([Paragraph(_md_inline(c), style_) for c in row])
                    t = Table(pdf_rows, colWidths=[col_w]*n_cols, repeatRows=1)
                    t.setStyle(TableStyle([
                        ("BACKGROUND",     (0,0), (-1,0),  C_NAVY),
                        ("ROWBACKGROUNDS", (0,1), (-1,-1), [white, C_LGRAY]),
                        ("GRID",           (0,0), (-1,-1), 0.5, C_MGRAY),
                        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
                        ("TOPPADDING",     (0,0), (-1,-1), 6),
                        ("BOTTOMPADDING",  (0,0), (-1,-1), 6),
                        ("LEFTPADDING",    (0,0), (-1,-1), 7),
                        ("RIGHTPADDING",   (0,0), (-1,-1), 7),
                    ]))
                    story.append(Spacer(1, 2*mm))
                    story.append(t)
                    story.append(Spacer(1, 3*mm))
                continue

            # Bullet
            if re.match(r'^[-*] ', line):
                txt = line[2:].strip()
                is_ref_item = any(txt.startswith(p) for p in
                                  ("[Web]", "[LG]", "[CATL]", "[Market"))
                if is_ref_item:
                    story.append(Paragraph(f"• {_md_inline(txt)}", s_ref))
                else:
                    story.append(Paragraph(f"• {_md_inline(txt)}", s_bullet))
                i += 1
                continue

            # HR
            if line.strip() in ("---", "***", "___"):
                story.append(HRFlowable(
                    width="100%", thickness=0.5, color=C_MGRAY, spaceAfter=3,
                ))
                i += 1
                continue

            # Empty
            if not line.strip():
                story.append(Spacer(1, 2*mm))
                i += 1
                continue

            # Normal paragraph
            story.append(Paragraph(_md_inline(line), s_body))
            i += 1

        canvas_factory = _make_canvas_factory(fn, fn_bold)
        doc.build(story, canvasmaker=canvas_factory)
        logger.info(f"PDF 생성: {pdf_path}")

    except Exception as e:
        logger.warning(f"PDF 변환 실패: {e}", exc_info=True)


# ── 보고서 작성 노드 ───────────────────────────────────────────
def report_writer_node(state: AgentState) -> dict:
    logger.info("[Report Writer] 보고서 생성 시작")
    try:
        llm = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            max_tokens=4096,
        )

        # 보고서 본문 생성
        report = llm.invoke([
            SystemMessage(content="당신은 배터리 산업 전문 보고서 작성자입니다. 마크다운으로 작성하세요."),
            HumanMessage(content=REPORT_PROMPT.format(
                market_background=_clean(state.get("market_background", ""), 3000),
                lg_strategy      =_clean(state.get("lg_rag_result",     ""), 5000),
                catl_strategy    =_clean(state.get("catl_rag_result",   ""), 5000),
                comparison       =_clean(state.get("comparison",        ""), 4000),
                implications     =_clean(state.get("implications",      ""), 3000),
                swot_lg          =_clean(state.get("swot_lg",  {})),
                swot_catl        =_clean(state.get("swot_catl",{})),
                feedback         =_clean(state.get("revision_feedback", "")),
            )),
        ]).content

        # SUMMARY 생성
        summary = llm.invoke([
            SystemMessage(content="250단어 이내로 핵심 인사이트를 요약하세요."),
            HumanMessage(content=SUMMARY_PROMPT.format(report=_clean(report, 5000))),
        ]).content

        # LLM이 생성한 SUMMARY 섹션 교체 (중복 방지)
        if "# SUMMARY" in report:
            next_sec = re.search(r'\n# [^#]', report[report.index("# SUMMARY") + 9:])
            if next_sec:
                rest = report[report.index("# SUMMARY") + 9 + next_sec.start() + 1:]
                report = f"# SUMMARY\n\n{summary}\n\n{rest}"
            else:
                report = f"# SUMMARY\n\n{summary}"
        else:
            report = f"# SUMMARY\n\n{summary}\n\n{report}"

        # LLM이 생성한 REFERENCE 제거 후 실제 참조 목록 추가
        if "# REFERENCE" in report:
            report = report[:report.index("# REFERENCE")].rstrip()
        refs = _build_references(state, llm)
        if refs:
            report += "\n\n" + refs

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
