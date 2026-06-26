from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                  TableStyle, Image as RLImage, HRFlowable)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io
import base64
import os
import tempfile
from datetime import datetime

PRIMARY_COLOR = colors.HexColor('#1565C0')
ACCENT_COLOR = colors.HexColor('#4FC3F7')
DARK_TEXT = colors.HexColor('#1A1A2E')
GRAY_TEXT = colors.HexColor('#6B7280')

SEVERITY_COLORS = {
    'Low': colors.HexColor('#16A34A'),
    'Moderate': colors.HexColor('#D97706'),
    'High': colors.HexColor('#DC2626')
}

def generate_pdf_report(report_data):
    tmp_dir = tempfile.gettempdir()
    pdf_path = os.path.join(tmp_dir, f"report_{report_data['id']}.pdf")

    doc = SimpleDocTemplate(
        pdf_path, pagesize=A4,
        topMargin=15*mm, bottomMargin=15*mm,
        leftMargin=18*mm, rightMargin=18*mm
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Title'],
        fontSize=22, textColor=PRIMARY_COLOR, spaceAfter=2, alignment=TA_LEFT, fontName='Helvetica-Bold')
    subtitle_style = ParagraphStyle('SubtitleStyle', parent=styles['Normal'],
        fontSize=10, textColor=GRAY_TEXT, spaceAfter=10)
    section_style = ParagraphStyle('SectionStyle', parent=styles['Heading2'],
        fontSize=13, textColor=DARK_TEXT, spaceBefore=14, spaceAfter=8, fontName='Helvetica-Bold')
    label_style = ParagraphStyle('LabelStyle', parent=styles['Normal'],
        fontSize=9, textColor=GRAY_TEXT, fontName='Helvetica')
    value_style = ParagraphStyle('ValueStyle', parent=styles['Normal'],
        fontSize=11, textColor=DARK_TEXT, fontName='Helvetica-Bold', spaceAfter=8)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'],
        fontSize=10, textColor=DARK_TEXT, leading=15)
    disclaimer_style = ParagraphStyle('DisclaimerStyle', parent=styles['Normal'],
        fontSize=8, textColor=GRAY_TEXT, leading=12)

    story = []

    # ── Header ──
    story.append(Paragraph("DrishtiAI", title_style))
    story.append(Paragraph("AI-Powered Retinal Disease Detection &amp; Explainable Diagnosis System", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.2, color=PRIMARY_COLOR, spaceAfter=14))

    # ── Patient Info ──
    story.append(Paragraph("PATIENT INFORMATION", section_style))

    created_dt = report_data.get('created_at', '')
    try:
        created_fmt = datetime.fromisoformat(created_dt).strftime('%d %B %Y, %I:%M %p')
    except Exception:
        created_fmt = created_dt

    patient_table_data = [
        [Paragraph("Patient Name", label_style), Paragraph(report_data.get('patient_name', '—'), value_style)],
        [Paragraph("Examined By", label_style), Paragraph(f"{report_data.get('role','—')}", value_style)],
        [Paragraph("Institution", label_style), Paragraph(report_data.get('hospital', '—'), value_style)],
        [Paragraph("Report Date", label_style), Paragraph(created_fmt, value_style)],
        [Paragraph("Report ID", label_style), Paragraph(report_data.get('id', '—')[:8].upper(), value_style)],
    ]
    patient_table = Table(patient_table_data, colWidths=[110, 350])
    patient_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('TOPPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 8))

    # ── Diagnosis Summary Box ──
    story.append(Paragraph("DIAGNOSIS SUMMARY", section_style))

    severity = report_data.get('severity', 'Low')
    severity_color = SEVERITY_COLORS.get(severity, colors.gray)

    diagnosis_table_data = [
        [Paragraph("DETECTED CONDITION", label_style), Paragraph("CONFIDENCE SCORE", label_style)],
        [Paragraph(report_data.get('disease', '—'),
                    ParagraphStyle('Disease', parent=value_style, fontSize=16, textColor=PRIMARY_COLOR)),
         Paragraph(f"{report_data.get('confidence', 0)}%",
                    ParagraphStyle('Conf', parent=value_style, fontSize=16, textColor=PRIMARY_COLOR))],
        [Paragraph("SEVERITY LEVEL", label_style), Paragraph("", label_style)],
        [Paragraph(severity, ParagraphStyle('Sev', parent=value_style, fontSize=14, textColor=severity_color)), ''],
    ]
    diagnosis_table = Table(diagnosis_table_data, colWidths=[230, 230])
    diagnosis_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
        ('BOX', (0,0), (-1,-1), 0.75, colors.HexColor('#E2E8F0')),
        ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 14),
        ('SPAN', (0,2), (1,2)),
    ]))
    story.append(diagnosis_table)
    story.append(Spacer(1, 10))

    # ── Clinical Recommendation ──
    story.append(Paragraph("CLINICAL RECOMMENDATION", section_style))
    story.append(Paragraph(report_data.get('recommendation', '—'), body_style))
    story.append(Spacer(1, 10))

    # ── Images ──
    story.append(Paragraph("RETINAL IMAGE ANALYSIS", section_style))

    try:
        orig_bytes = base64.b64decode(report_data['original_image'])
        orig_path = os.path.join(tmp_dir, f"orig_{report_data['id']}.png")
        with open(orig_path, 'wb') as f:
            f.write(orig_bytes)

        heatmap_bytes = base64.b64decode(report_data['heatmap'])
        heatmap_path = os.path.join(tmp_dir, f"heat_{report_data['id']}.png")
        with open(heatmap_path, 'wb') as f:
            f.write(heatmap_bytes)

        img1 = RLImage(orig_path, width=200, height=200)
        img2 = RLImage(heatmap_path, width=200, height=200)

        caption_style = ParagraphStyle('Caption', parent=styles['Normal'],
            fontSize=8.5, textColor=GRAY_TEXT, alignment=TA_CENTER, spaceBefore=4)

        img_table = Table([
            [img1, img2],
            [Paragraph("Original Fundus Image", caption_style),
             Paragraph("Grad-CAM Explainability Heatmap", caption_style)]
        ], colWidths=[230, 230])
        img_table.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ]))
        story.append(img_table)
    except Exception as e:
        story.append(Paragraph(f"(Images unavailable: {e})", body_style))

    story.append(Spacer(1, 14))

    # ── Disclaimer ──
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor('#E2E8F0'), spaceAfter=8))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an AI-assisted screening tool (DrishtiAI) and is intended "
        "to support, not replace, clinical judgment. All findings must be reviewed and confirmed by a qualified "
        "ophthalmologist before any diagnostic or treatment decisions are made.",
        disclaimer_style
    ))
    story.append(Spacer(1, 4))
    story.append(Paragraph("Generated by DrishtiAI &mdash; Retinal Disease Detection System", disclaimer_style))

    doc.build(story)
    return pdf_path