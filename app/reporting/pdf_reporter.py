"""Industrial Quality Assurance PDF Report Generator."""

from pathlib import Path
from typing import Union
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

from app.inspection.decision_engine import StandardizedInspectionVerdict


class PDFReporter:
    """Generates professional, publication-quality AOI inspection PDF reports."""

    @classmethod
    def generate_pdf_report(
        cls,
        verdict: StandardizedInspectionVerdict,
        output_path: Union[str, Path]
    ) -> Path:
        """Compile complete inspection verdict into an audit PDF."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)

        doc = SimpleDocTemplate(
            str(p),
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "DocTitle",
            parent=styles["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#1e293b")
        )
        subtitle_style = ParagraphStyle(
            "DocSubTitle",
            parent=styles["Normal"],
            fontSize=10,
            leading=14,
            textColor=colors.HexColor("#64748b")
        )
        heading2_style = ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#334155")
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#334155")
        )

        story = []

        # Header Title
        story.append(Paragraph("AUTOMATED OPTICAL INSPECTION REPORT", title_style))
        story.append(Paragraph("Intelligent PCB Defect Inspection System | Phase 1 Classical Computer Vision", subtitle_style))
        story.append(Spacer(1, 10))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=12))

        # Status Badge Table
        badge_color = colors.HexColor("#16a34a") if verdict.decision == "PASS" else (
            colors.HexColor("#dc2626") if verdict.decision == "FAIL" else colors.HexColor("#d97706")
        )
        verdict_badge_text = f"<b><font size='14' color='white'>VERDICT: {verdict.decision}</font></b>"
        badge_data = [[Paragraph(verdict_badge_text, ParagraphStyle("B", alignment=1))]]
        badge_table = Table(badge_data, colWidths=[540])
        badge_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), badge_color),
            ("TOPPADDING", (0, 0), (-1, -1), 8),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(badge_table)
        story.append(Spacer(1, 14))

        # Metadata Table
        meta_data = [
            [Paragraph("<b>Inspection ID:</b>", body_style), Paragraph(verdict.inspection_id, body_style),
             Paragraph("<b>Domain:</b>", body_style), Paragraph(verdict.inspection_type, body_style)],
            [Paragraph("<b>Timestamp (UTC):</b>", body_style), Paragraph(verdict.timestamp, body_style),
             Paragraph("<b>Processing Latency:</b>", body_style), Paragraph(f"{verdict.processing_time_ms:.1f} ms", body_style)],
            [Paragraph("<b>Decision Reason:</b>", body_style), Paragraph(verdict.decision_reason, body_style),
             Paragraph("<b>Total Defects / Anomalies:</b>", body_style), Paragraph(str(verdict.total_defect_count), body_style)],
        ]
        meta_table = Table(meta_data, colWidths=[110, 160, 130, 140])
        meta_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 14))

        # Defect Summary Breakdown Table
        story.append(Paragraph("Defect Breakdown by Category", heading2_style))
        story.append(Spacer(1, 6))

        summary_rows = [["Defect Category", "Count", "Classification Policy", "Resolution Action"]]
        if verdict.defect_summary:
            for cat, cnt in verdict.defect_summary.items():
                if cnt > 0:
                    action = "Rework / Scrapped" if cat in ["open", "short", "MISSING"] else "Review / Pass with concession"
                    summary_rows.append([cat.replace("_", " ").title(), str(cnt), "Automated Classical Rule", action])
        if len(summary_rows) == 1:
            summary_rows.append(["None", "0", "Zero defects detected", "Released to next assembly stage"])

        sum_table = Table(summary_rows, colWidths=[140, 60, 170, 170])
        sum_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(sum_table)
        story.append(Spacer(1, 14))

        # Itemized Defect Table (Top 12 items)
        if verdict.defects:
            story.append(Paragraph(f"Itemized Defect Register (Showing {min(12, len(verdict.defects))} of {len(verdict.defects)})", heading2_style))
            story.append(Spacer(1, 6))
            det_rows = [["ID", "Category", "Severity", "Bounding Box [x1,y1,x2,y2]", "Area (px)", "Conf"]]
            for d in verdict.defects[:12]:
                det_rows.append([
                    str(d.id),
                    d.category,
                    d.severity,
                    str(list(d.bounding_box)),
                    f"{d.area:.0f}",
                    f"{d.confidence:.2f}"
                ])
            det_table = Table(det_rows, colWidths=[30, 95, 75, 190, 80, 70])
            det_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(det_table)
            story.append(Spacer(1, 14))

        # Academic / Industrial Compliance Footer
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#94a3b8"), spaceAfter=8))
        footer_text = (
            "System: Intelligent PCB Defect Inspection System | Phase 1 Classical Computer Vision | "
            "Course: ACSC502 - Computer Vision (5th Semester) | Generated deterministically from real inspection telemetry."
        )
        story.append(Paragraph(footer_text, ParagraphStyle("Footer", parent=styles["Normal"], fontSize=7, textColor=colors.HexColor("#64748b"))))

        doc.build(story)
        return p
