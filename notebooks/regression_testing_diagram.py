"""
Minimal ACM TIST-style diagram of the HPMS regression testing pipeline.

Design goals:
- grayscale-first, print-friendly
- minimal visual decoration
- strong hierarchy through spacing and line weight
- no redundant legend
- portable output paths
"""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ── Typography ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ── Palette: grayscale + one muted accent ────────────────────────────────────
C_BG       = "#FFFFFF"
C_TEXT     = "#111111"
C_MUTED    = "#5A5A5A"
C_LINE     = "#2E2E2E"
C_BORDER   = "#6E6E6E"
C_FILL     = "#F5F5F5"
C_SUBFILL  = "#FAFAFA"
C_ACCENT   = "#4C566A"   # muted blue-gray, still print-safe
C_ARROW    = "#3A3A3A"

# ── Canvas ────────────────────────────────────────────────────────────────────
# close to single-column/small full-width journal figure proportions
FIG_W, FIG_H = 7.1, 9.0
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

LM = 0.55
RM = FIG_W - 0.55
W = RM - LM
CX = (LM + RM) / 2

# ── Helpers ───────────────────────────────────────────────────────────────────
def wrap(text, width):
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))

def box(x, y, w, h, title, subtitle=None, facecolor=C_FILL,
        edgecolor=C_BORDER, title_fs=8.3, sub_fs=7.2,
        lw=0.9, title_weight="bold", align="center"):
    """Simple journal-style rectangular box."""
    rect = Rectangle((x, y), w, h, linewidth=lw,
                     edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(rect)

    tx = x + w/2 if align == "center" else x + 0.10
    ha = "center" if align == "center" else "left"

    if subtitle:
        ax.text(tx, y + h*0.64, title, ha=ha, va="center",
                fontsize=title_fs, fontweight=title_weight,
                color=C_TEXT, zorder=3)
        ax.text(tx, y + h*0.34, subtitle, ha=ha, va="center",
                fontsize=sub_fs, color=C_MUTED, zorder=3)
    else:
        ax.text(tx, y + h/2, title, ha=ha, va="center",
                fontsize=title_fs, fontweight=title_weight,
                color=C_TEXT, zorder=3)

def bullet_box(x, y, w, h, title, subtitle, bullets,
               facecolor=C_SUBFILL, edgecolor=C_BORDER):
    rect = Rectangle((x, y), w, h, linewidth=0.85,
                     edgecolor=edgecolor, facecolor=facecolor, zorder=2)
    ax.add_patch(rect)

    ax.text(x + w/2, y + h - 0.18, title, ha="center", va="top",
            fontsize=8.0, fontweight="bold", color=C_TEXT, zorder=3)
    ax.text(x + w/2, y + h - 0.38, subtitle, ha="center", va="top",
            fontsize=7.0, color=C_MUTED, zorder=3)

    start_y = y + h - 0.66
    for i, item in enumerate(bullets):
        ax.text(x + 0.10, start_y - i*0.22, f"• {item}",
                ha="left", va="top", fontsize=6.9, color=C_TEXT, zorder=3)

def section_label(y, text):
    ax.text(LM, y, text, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color=C_ACCENT, zorder=4)

def down_arrow(x, y0, y1, lw=0.95):
    ax.annotate(
        "", xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=C_ARROW, mutation_scale=9),
        zorder=5
    )

def h_arrow(x0, x1, y, bidirectional=False, lw=0.9):
    style = "<|-|>" if bidirectional else "-|>"
    ax.annotate(
        "", xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle=style, lw=lw, color=C_ARROW, mutation_scale=8),
        zorder=5
    )

# ── Title ─────────────────────────────────────────────────────────────────────
ax.text(CX, 8.68, "Regression Testing Pipeline",
        ha="center", va="center", fontsize=13, fontweight="bold", color=C_TEXT)
ax.text(
    CX, 8.46,
    "Daily automated safety evaluation of AI psychiatric companion conversations",
    ha="center", va="center", fontsize=8.2, color=C_MUTED
)
ax.plot([LM, RM], [8.30, 8.30], color=C_LINE, lw=0.8, solid_capstyle="butt")

# ── Section 1: CI/CD ─────────────────────────────────────────────────────────
section_label(8.04, "1. CI/CD")
box(
    LM, 7.54, W, 0.38,
    title="GitHub Actions workflow: regression-test.yml",
    subtitle="Triggers: daily schedule (06:00 UTC) and manual dispatch; "
             "matrix: 2 rounds × 2 chat models = 4 generation jobs",
    facecolor=C_FILL
)

# ── Section 2: Input datasets ────────────────────────────────────────────────
down_arrow(CX, 7.54, 7.18)
section_label(6.98, "2. Input datasets")

s2_y, s2_h = 6.32, 0.56
gap = 0.16
bw = (W - gap) / 2

box(
    LM, s2_y, bw, s2_h,
    title="Standardized Safety Dataset",
    subtitle="Park et al. (2024)",#\n64,561 questions",
    facecolor=C_SUBFILL
)
box(
    LM + bw + gap, s2_y, bw, s2_h,
    title="Open-ended Dataset",
    subtitle="HuggingFace everyday-conversations-llama3.2-1k", #\n8,203 conversation starters",
    facecolor=C_SUBFILL
)

# ── Section 3: Conversation generation ───────────────────────────────────────
down_arrow(CX, s2_y, 5.92)
section_label(5.72, "3. Conversation generation")

ax.text(
    CX, 5.58,
    "Simulated multi-turn dialogue; output stored as dataset-round-{2,3}-{model}.json "
    "with Langfuse tracing",
    ha="center", va="center", fontsize=7.0, color=C_MUTED
)

s3_y, s3_h = 4.70, 0.66
gap3 = 0.12
bw3 = (W - 2*gap3) / 3

x1 = LM
x2 = LM + bw3 + gap3
x3 = LM + 2*(bw3 + gap3)

box(x1, s3_y, bw3, s3_h, "Companion agent", "GPT-4o / GPT-5", facecolor=C_SUBFILL)
box(x2, s3_y, bw3, s3_h, "In-silico processor", "Up to 5 turns / session", facecolor=C_SUBFILL)
box(x3, s3_y, bw3, s3_h, "Psychiatrist agent", "GPT-4o / GPT-5", facecolor=C_SUBFILL)

midy = s3_y + s3_h/2
h_arrow(x1 + bw3, x2, midy, bidirectional=True)
h_arrow(x2 + bw3, x3, midy, bidirectional=True)



# ── Section 4: Automated evaluation ──────────────────────────────────────────
down_arrow(CX, s3_y, 4.34)
section_label(4.14, "4. Automated safety evaluation")

s4_y, s4_h = 2.78, 1.12
gap4 = 0.12
bw4 = (W - 2*gap4) / 3

bullet_box(
    LM, s4_y, bw4, s4_h,
    title="LlamaGuard",
    subtitle="Llama-Guard-4-12B",
    bullets=[
        "14 safety categories (S1–S14)",
        "Per-message classification",
        #"Safe / unsafe label",
    ],
)

bullet_box(
    LM + bw4 + gap4, s4_y, bw4, s4_h,
    title="OpenAI moderation",
    subtitle="omni-moderation-2024-09-26",
    bullets=[
        "21 moderation categories",
        "Per-message scores",
        #"Flagged category list",
    ],
)

bullet_box(
    LM + 2*(bw4 + gap4), s4_y, bw4, s4_h,
    title="LLM-as-judge",
    subtitle="OpenAI Batch API",
    bullets=[
        "Structured rating rubric",
        "Batch processing (JSONL)",
        #"Numeric quality scores",
    ],
)

# ax.text(
#     CX, 2.56,
#     "Sequential execution: LlamaGuard → OpenAI Moderation → LLM-as-Judge",
#     ha="center", va="center", fontsize=7.0, color=C_MUTED
# )

# ── Section 5: Storage ───────────────────────────────────────────────────────
down_arrow(CX, s4_y, 2.18)
section_label(1.98, "5. Result storage")

s5_y, s5_h = 1.46, 0.42
bw5 = (W - gap) / 2

box(
    LM, s5_y, bw5, s5_h,
    title="GitHub artifacts",
    subtitle="Per-run snapshots; 7-day retention",
    facecolor=C_SUBFILL
)
box(
    LM + bw5 + gap, s5_y, bw5, s5_h,
    title="Google Drive (shared drive)",
    subtitle="Date-partitioned long-term archive",
    facecolor=C_SUBFILL
)

# ── Section 6: Analysis ──────────────────────────────────────────────────────
down_arrow(CX, s5_y, 1.18)
section_label(0.98, "6. Post-processing and analysis")

s6_y, s6_h = 0.18, 0.58
gap6 = 0.08
metrics = [
    ("Safety trends", "agg_reg_test.py"),
    ("Moderation agreement", "reg_test_moderation_agreement.py"),
    ("Category breakdown", "reg_test_categories_plots.py"),
    ("Diversity metrics", "self_bleu.py /\nunique_n_grams.py"),
    ("LLM judge ratings", "reg_test_llm_judge_\nratings_boxplot.py"),
    ("LaTeX tables", "reg_test_table.py"),
]
bw6 = (W - gap6*(len(metrics)-1)) / len(metrics)

for i, (title, subtitle) in enumerate(metrics):
    xi = LM + i*(bw6 + gap6)
    box(
        xi, s6_y, bw6, s6_h,
        title=wrap(title, 16),
        subtitle=subtitle,
        facecolor=C_SUBFILL,
        title_fs=7.2,
        sub_fs=6.2,
        lw=0.8
    )

# ── Save ──────────────────────────────────────────────────────────────────────
out_dir = Path.cwd()
out_pdf = out_dir / "regression_testing_pipeline_acm_tist.pdf"
out_png = out_dir / "regression_testing_pipeline_acm_tist.png"

fig.savefig(out_pdf, bbox_inches="tight", facecolor=C_BG)
fig.savefig(out_png, bbox_inches="tight", facecolor=C_BG)
print(f"Saved:\n  {out_pdf}\n  {out_png}")