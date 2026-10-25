"""report.py - builds the downloadable PDF screening report."""
import io
from datetime import datetime

import numpy as np
from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image as RLImage
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from dr_core import GRADES, LESIONS

DISCLAIMER = ("This report is produced by a student research prototype. It is NOT a medical device and must "
              "not be used for diagnosis or treatment decisions. Always consult a qualified clinician.")


def _img(arr, width_mm):
    im = Image.fromarray(np.asarray(arr, dtype=np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    buf.seek(0)
    w, h = im.size
    return RLImage(buf, width=width_mm * mm, height=width_mm * mm * h / w)


def build_pdf(patient, res, thr_burden, metabolic, fusion, images):
    """patient: dict(name, age); res: analyse() output; thr_burden: list[4] %; images: dict of RGB arrays."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=14 * mm, bottomMargin=14 * mm, title="Retina screening report")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], fontSize=20, textColor=colors.HexColor("#0f766e"), alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=13, textColor=colors.HexColor("#1e293b"), spaceBefore=10)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.HexColor("#64748b"))
    tier_col = colors.HexColor(fusion["color"])

    s = [Paragraph("RetinaGuard - Diabetic Retinopathy Screening Report", h1),
         Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M}", small), Spacer(1, 4)]

    info = [["Patient", patient.get("name") or "Not provided", "Age", str(patient.get("age", "-"))]]
    t = Table(info, colWidths=[22 * mm, 70 * mm, 18 * mm, 60 * mm])
    t.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 9.5), ("TEXTCOLOR", (0, 0), (0, 0), colors.grey),
                           ("TEXTCOLOR", (2, 0), (2, 0), colors.grey),
                           ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1"))]))
    s += [t, Spacer(1, 8)]

    banner = Table([[Paragraph(f"<font color='white'><b>Overall risk: {fusion['name'].upper()}</b></font>", body),
                     Paragraph(f"<font color='white'>{fusion['action']} <b>({fusion['when']})</b></font>", body)]],
                   colWidths=[50 * mm, 126 * mm])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), tier_col), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                                ("TOPPADDING", (0, 0), (-1, -1), 8), ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    s += [banner, Spacer(1, 4)]
    for r in fusion["reasons"]:
        s.append(Paragraph(f"&bull; {r}", body))

    s.append(Paragraph("1. Retinal image analysis", h2))
    g = res["grade"]
    s.append(Paragraph(f"<b>DR grade {g} - {GRADES[g]}</b> (confidence {res['confidence'] * 100:.0f}%). "
                       f"Image quality: {res['quality']['label']}.", body))
    probs = [["Grade"] + [f"{i}: {n}" for i, n in enumerate(GRADES)],
             ["Probability"] + [f"{p * 100:.1f}%" for p in res["probs"]]]
    pt = Table(probs, colWidths=[24 * mm] + [30 * mm] * 5)
    pt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9"))]))
    s += [Spacer(1, 4), pt, Spacer(1, 6)]

    cells = []
    for key, cap in (("original", "Original"), ("cam", "Grad-CAM"), ("lesions", "Lesion map"), ("landmarks", "Landmarks")):
        if key in images:
            cells.append([_img(images[key], 41), Paragraph(cap, small)])
    if cells:
        row1 = Table([[c[0] for c in cells], [c[1] for c in cells]], colWidths=[43 * mm] * len(cells))
        s += [row1]
    if thr_burden:
        rows = [["Lesion type", "Area of retina covered"]] + [[LESIONS[i], f"{thr_burden[i]:.2f}%"] for i in range(4)]
        lt = Table(rows, colWidths=[60 * mm, 50 * mm])
        lt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9"))]))
        s += [Spacer(1, 6), lt]

    s.append(Paragraph("2. Metabolic risk", h2))
    s.append(Paragraph(f"<b>Metabolic risk: {metabolic['level']}</b> (score {metabolic['score']:.0f}/100)", body))
    rows = [["Factor", "Value", "Status", "Weight"]] + [[f["name"], f["value"], f["label"], f["weight"]] for f in metabolic["factors"]]
    mt = Table(rows, colWidths=[50 * mm, 45 * mm, 45 * mm, 30 * mm])
    mt.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f1f5f9"))]))
    s += [Spacer(1, 4), mt, Spacer(1, 4)]
    for a in metabolic["advice"]:
        s.append(Paragraph(f"&bull; {a}", body))

    s += [Spacer(1, 12), Paragraph(DISCLAIMER, small)]
    doc.build(s)
    return buf.getvalue()
