"""Render the Cardiometabolic Digital Twin architecture diagram to PDF and PNG.

Pure-matplotlib renderer with an explicit coordinate grid so every layer box is
identically sized and centered, and every connector is a perfectly straight,
centered vertical line. Run:  python docs/make_architecture_diagram.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, FancyArrow

# ---- Alignment grid (data coordinates; 1 unit = 1 px) ----
W, H = 1100, 1360
CENTER_X = 550.0           # single vertical axis for all connectors
BOX_X = 170.0              # left edge of every layer box
BOX_W = 760.0              # width of every layer box
BOX_H = 150.0              # height of every layer box
GAP = 56.0                 # vertical gap between boxes
BADGE_W = 150.0            # colored number badge width

LAYERS = [
    ("1", "Data Generation", "#3b82f6", "#eff6ff", [
        "Synthetic Synthea-style EHR cohort — 300 T2DM patients",
        "Bergman Minimal Model ODE + Dalla Man gut absorption kinetics",
        "300 x 14-day wearable streams @ 5-min (CGM, HR, HRV, steps, sleep)",
    ]),
    ("2", "Feature Pipeline", "#14b8a6", "#f0fdfa", [
        "4-hour sliding window (48 readings), stepped every 30 min",
        "47 features = 18 static EHR + 29 dynamic wearable signals",
        "Patient-level split — 198,000 samples, zero data leakage",
    ]),
    ("3", "Machine Learning & Explainability", "#ef4444", "#fef2f2", [
        "Fused XGBoost regressor — 2-hour glucose forecast (R2 = 0.805)",
        "Class-weighted adverse-event classifier (AUROC = 0.976)",
        "SHAP TreeExplainer — per-prediction clinical attributions",
    ]),
    ("4", "Digital Twin Engine", "#f59e0b", "#fffbeb", [
        "Stateful per-patient twin initialized from the EHR profile",
        "predict() — 2h forecast, risk level, top-5 SHAP factors",
        "simulate() — deterministic Bergman ODE What-If counterfactuals",
    ]),
    ("5", "FastAPI REST Backend", "#8b5cf6", "#faf5ff", [
        "Async API with per-patient twin caching, served on port 8000",
        "/patients /patient/{id} /state /timeline /predict /simulate",
        "Serves the dashboard and OpenAPI (Swagger) documentation",
    ]),
    ("6", "Clinician Dashboard", "#0ea5e9", "#f8fafc", [
        "Patient 360 profile  -  SVG 2-hour adverse-risk gauge",
        "Dual-axis CGM / heart-rate timeline with 2h forecast overlay",
        "SHAP attribution bars  -  What-If intervention simulator (live)",
    ]),
]

fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()          # top-down layout
ax.axis("off")

# ---- Title band ----
ax.add_patch(FancyBboxPatch((BOX_X, 24), BOX_W, 78,
             boxstyle="round,pad=0,rounding_size=14",
             linewidth=0, facecolor="#0f172a"))
ax.text(CENTER_X, 58, "Cardiometabolic Digital Twin",
        ha="center", va="center", fontsize=20, fontweight="bold", color="white")
ax.text(CENTER_X, 86,
        "Predictive Glycemic Excursion & Counterfactual Intervention System for Type 2 Diabetes",
        ha="center", va="center", fontsize=10.5, color="#94a3b8")

top0 = 146.0

def box_top(i):
    return top0 + i * (BOX_H + GAP)

for i, (num, title, color, fill, lines) in enumerate(LAYERS):
    top = box_top(i)
    # Full box
    ax.add_patch(FancyBboxPatch((BOX_X, top), BOX_W, BOX_H,
                 boxstyle="round,pad=0,rounding_size=14",
                 linewidth=2, edgecolor=color, facecolor=fill))
    # Colored number badge (left)
    ax.add_patch(FancyBboxPatch((BOX_X, top), BADGE_W, BOX_H,
                 boxstyle="round,pad=0,rounding_size=14",
                 linewidth=0, facecolor=color))
    # Square off the badge's right corners so it reads as a flush band
    ax.add_patch(Rectangle((BOX_X + BADGE_W - 16, top), 16, BOX_H,
                 linewidth=0, facecolor=color))
    bx = BOX_X + BADGE_W / 2.0
    ax.text(bx, top + BOX_H / 2.0 - 8, num, ha="center", va="center",
            fontsize=34, fontweight="bold", color="white")
    ax.text(bx, top + BOX_H / 2.0 + 26, "LAYER", ha="center", va="center",
            fontsize=10, fontweight="bold", color="white")
    # Title + description, left-aligned on a fixed text column
    tx = BOX_X + BADGE_W + 26
    ax.text(tx, top + 34, title, ha="left", va="center",
            fontsize=16, fontweight="bold", color="#0f172a")
    for j, line in enumerate(lines):
        ax.text(tx, top + 70 + j * 24, line, ha="left", va="center",
                fontsize=11.5, color="#334155")
    # Straight centered connector to the next box
    if i < len(LAYERS) - 1:
        y_start = top + BOX_H
        y_end = box_top(i + 1)
        ax.add_patch(FancyArrow(
            CENTER_X, y_start, 0, (y_end - y_start) - 6,
            width=0.6, head_width=12, head_length=10,
            length_includes_head=True, color="#334155"))

out_pdf = "docs/architecture_diagram.pdf"
out_png = "docs/architecture_diagram.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=150)
plt.close(fig)
print(f"Saved {out_pdf} and {out_png}")
