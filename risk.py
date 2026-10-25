"""
risk.py - rule-based metabolic risk + decision-level fusion with the retinal DR grade.

IMPORTANT (state this in your report): no public dataset links retina images with HbA1c / BP,
so the two parts are NOT jointly trained. They are combined by transparent clinical rules.
Thresholds follow common guideline cut-offs (ADA HbA1c bands, ACC/AHA blood-pressure stages)
and are for a student prototype only - not for clinical use.
"""

TIERS = [
    {"name": "Low", "color": "#16a34a", "icon": "✅"},
    {"name": "Elevated", "color": "#ca8a04", "icon": "👀"},
    {"name": "High", "color": "#ea580c", "icon": "⚠️"},
    {"name": "Urgent", "color": "#dc2626", "icon": "🚨"},
]

GRADE_ACTIONS = [
    ("Continue routine annual eye screening.", "within 12 months"),
    ("Repeat retinal screening and review glycaemic control.", "within 6-12 months"),
    ("Refer to an ophthalmologist for a full dilated eye exam.", "within 3-6 months"),
    ("Refer to an ophthalmologist promptly - severe non-proliferative changes.", "within 2-4 weeks"),
    ("Urgent ophthalmology referral - proliferative disease, risk of vision loss.", "within 1 week"),
]
GRADE_TIER = [0, 1, 2, 3, 3]            # DR grade -> starting tier


def _hba1c_points(v):
    return 0 if v < 5.7 else 1 if v < 6.5 else 2 if v < 8.0 else 3 if v < 10.0 else 4


def _bp_points(s, d):
    if s < 120 and d < 80:
        return 0
    if s < 130 and d < 80:
        return 1
    if s < 140 and d < 90:
        return 2
    if s < 160 and d < 100:
        return 3
    return 4


def _duration_points(y):
    return 0 if y < 5 else 1 if y < 10 else 2 if y < 15 else 3 if y < 20 else 4


def _label(pts):
    return ["Normal", "Slightly raised", "Raised", "High", "Very high"][pts]


def metabolic_risk(hba1c, systolic, diastolic, duration_years):
    """Returns score 0-100, level (Low/Medium/High) and the per-factor breakdown."""
    pts = {"hba1c": _hba1c_points(hba1c), "bp": _bp_points(systolic, diastolic),
           "duration": _duration_points(duration_years)}
    weights = {"hba1c": 0.45, "bp": 0.30, "duration": 0.25}
    score = round(100 * sum(weights[k] * pts[k] / 4 for k in pts), 1)
    level = "Low" if score < 30 else "Medium" if score < 60 else "High"
    color = {"Low": "#16a34a", "Medium": "#ca8a04", "High": "#dc2626"}[level]
    factors = [
        {"name": "HbA1c", "value": f"{hba1c:.1f} %", "pts": pts["hba1c"], "label": _label(pts["hba1c"]),
         "weight": "45%"},
        {"name": "Blood pressure", "value": f"{int(systolic)}/{int(diastolic)} mmHg", "pts": pts["bp"],
         "label": _label(pts["bp"]), "weight": "30%"},
        {"name": "Diabetes duration", "value": f"{duration_years:g} yrs", "pts": pts["duration"],
         "label": _label(pts["duration"]), "weight": "25%"},
    ]
    advice = []
    if hba1c >= 8.0:
        advice.append("HbA1c is well above target - discuss treatment intensification with the physician.")
    elif hba1c >= 6.5:
        advice.append("HbA1c is in the diabetic range - keep monitoring every 3 months.")
    if systolic >= 140 or diastolic >= 90:
        advice.append("Blood pressure is raised - BP control slows retinopathy progression.")
    if duration_years >= 10:
        advice.append("Long diabetes duration raises lifetime retinopathy risk - never skip yearly eye checks.")
    if not advice:
        advice.append("Metabolic markers look well controlled - maintain current lifestyle and follow-up.")
    return {"score": score, "level": level, "color": color, "factors": factors, "advice": advice}


def fuse(dr_grade, dr_confidence, metabolic):
    """Combine retinal grade + metabolic score into one tiered recommendation."""
    tier = GRADE_TIER[dr_grade]
    reasons = [f"Retina model graded the image as grade {dr_grade} ({dr_confidence * 100:.0f}% confidence)."]
    if metabolic["score"] >= 60 and dr_grade >= 1 and tier < 3:
        tier += 1
        reasons.append("Poor metabolic control (score >= 60) with existing retinopathy -> risk tier raised by one level.")
    elif metabolic["score"] >= 60:
        reasons.append("Metabolic control is poor even though no retinopathy was detected - tighten control and rescreen sooner.")
    action, when = GRADE_ACTIONS[dr_grade]
    if tier > GRADE_TIER[dr_grade]:
        when = {1: "within 6 months", 2: "within 2-3 months", 3: "within 2 weeks"}[tier] if dr_grade < 3 else when
    if dr_confidence < 0.55:
        reasons.append("Model confidence is low - have a clinician review the image directly.")
    dr_index = [5, 25, 50, 80, 100][dr_grade]
    combined = round(0.65 * dr_index + 0.35 * metabolic["score"], 1)
    return {"tier": tier, **TIERS[tier], "action": action, "when": when, "reasons": reasons,
            "combined_score": combined}
