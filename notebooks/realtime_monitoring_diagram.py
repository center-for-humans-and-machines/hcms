"""
Minimal ACM TIST-style diagram of the HPMS real-time monitoring pipeline
(hpms-dashboard repository).

Design goals:
- grayscale-first, print-friendly
- minimal visual decoration
- strong hierarchy through spacing and line weight
- no redundant legend
- portable output paths
"""

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

# ── Palette ───────────────────────────────────────────────────────────────────
C_BG      = "#FFFFFF"
C_TEXT    = "#111111"
C_MUTED   = "#5A5A5A"
C_LINE    = "#2E2E2E"
C_BORDER  = "#6E6E6E"
C_FILL    = "#F5F5F5"
C_SUBFILL = "#FAFAFA"
C_ACCENT  = "#4C566A"
C_ARROW   = "#3A3A3A"
C_EXT     = "#EBEBEB"   # external / upstream boxes

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 7.1, 10.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

LM = 0.55
RM = FIG_W - 0.55
W  = RM - LM
CX = (LM + RM) / 2

# ── Helpers ───────────────────────────────────────────────────────────────────
def box(x, y, w, h, title, subtitle=None,
        facecolor=C_FILL, edgecolor=C_BORDER,
        title_fs=8.3, sub_fs=7.2, lw=0.9, title_weight="bold"):
    """Rectangular box with optional two-line label."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw,
                            edgecolor=edgecolor, facecolor=facecolor, zorder=2))
    if subtitle:
        ax.text(x + w/2, y + h*0.64, title,
                ha="center", va="center",
                fontsize=title_fs, fontweight=title_weight, color=C_TEXT, zorder=3)
        ax.text(x + w/2, y + h*0.30, subtitle,
                ha="center", va="center",
                fontsize=sub_fs, color=C_MUTED, style="italic", zorder=3)
    else:
        ax.text(x + w/2, y + h/2, title,
                ha="center", va="center",
                fontsize=title_fs, fontweight=title_weight, color=C_TEXT, zorder=3)


def bullet_box(x, y, w, h, title, subtitle, bullets,
               facecolor=C_SUBFILL, edgecolor=C_BORDER, lw=0.85):
    """Box with title, italic subtitle, and bullet list."""
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw,
                            edgecolor=edgecolor, facecolor=facecolor, zorder=2))
    ax.text(x + w/2, y + h - 0.19, title,
            ha="center", va="top",
            fontsize=8.0, fontweight="bold", color=C_TEXT, zorder=3)
    ax.text(x + w/2, y + h - 0.39, subtitle,
            ha="center", va="top",
            fontsize=7.0, color=C_MUTED, style="italic", zorder=3)
    for i, item in enumerate(bullets):
        ax.text(x + 0.13, y + h - 0.64 - i * 0.225,
                f"• {item}", ha="left", va="top",
                fontsize=6.9, color=C_TEXT, zorder=3)


def section_label(y, text):
    """Bold colored section heading, anchored at its baseline."""
    ax.text(LM, y, text, ha="left", va="bottom",
            fontsize=8.8, fontweight="bold", color=C_ACCENT, zorder=4)


def italicnote(y, text, fs=7.1):
    """Centered italic annotation line."""
    ax.text(CX, y, text, ha="center", va="center",
            fontsize=fs, color=C_MUTED, style="italic", zorder=4)


def down_arrow(x, y_top, y_bot, lw=0.95):
    ax.annotate("", xy=(x, y_bot), xytext=(x, y_top),
                arrowprops=dict(arrowstyle="-|>", lw=lw,
                                color=C_ARROW, mutation_scale=9),
                zorder=5)


def h_bidiarrow(x0, x1, y, lw=0.9):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<|-|>", lw=lw,
                                color=C_ARROW, mutation_scale=8),
                zorder=5)


# ════════════════════════════════════════════════════════════════════════════
# COORDINATES  (y increases upward; origin at bottom-left)
#
#  All "box y" values are the BOTTOM edge of the box.
#  Arrows connect box bottoms to the target y below.
# ════════════════════════════════════════════════════════════════════════════

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(CX, 10.08, "Real-Time Monitoring Pipeline",
        ha="center", va="center",
        fontsize=13, fontweight="bold", color=C_TEXT)
ax.text(CX, 9.84,
        "Live safety review of AI psychiatric companion conversations  (hpms-dashboard)",
        ha="center", va="center", fontsize=8.2, color=C_MUTED)
ax.plot([LM, RM], [9.68, 9.68], color=C_LINE, lw=0.8, solid_capstyle="butt")

# ════════════════════════════════════════════════════════════════════════════
# 1. Upstream sources
# ════════════════════════════════════════════════════════════════════════════
section_label(9.42, "1. Upstream sources  (external to dashboard)")

s1_y, s1_h = 8.78, 0.52
gap1 = 0.20
bw1  = (W - gap1) / 2

box(LM,              s1_y, bw1, s1_h,
    "Participant conversations",
    "SimpleChat / Android app  ->  experiment platform",
    facecolor=C_EXT, lw=0.8)

box(LM + bw1 + gap1, s1_y, bw1, s1_h,
    "Automated safety checks",
    "OpenAI Moderation API  +  LlamaGuard  (hpms monitoring)",
    facecolor=C_EXT, lw=0.8)

# ════════════════════════════════════════════════════════════════════════════
# 2. Shared database
# ════════════════════════════════════════════════════════════════════════════
down_arrow(CX, s1_y, 8.42)          # arrow from bottom of S1 boxes downward
section_label(8.20, "2. Shared database")

s2_y, s2_h = 7.56, 0.50
box(LM, s2_y, W, s2_h,
    "MongoDB Atlas  (replica set)",
    "Conversations collection: messages[], reviewer_flags[], flagged  "
    "|  Dashboard writes: assigned_messages[], reviewed_messages[], flags",
    facecolor=C_FILL)

# ════════════════════════════════════════════════════════════════════════════
# 3. Real-time change detection
# ════════════════════════════════════════════════════════════════════════════
down_arrow(CX, s2_y, 7.18)
section_label(6.96, "3. Real-time change detection")

italicnote(6.80,
    "MongoDB change stream watches Conversations collection; "
    "'opened_by' updates are filtered to prevent refresh loops")

s3_y, s3_h = 6.06, 0.62
gap3 = 0.20
bw3  = (W - gap3) / 2

box(LM,              s3_y, bw3, s3_h,
    "LLM flag detection",
    "reviewer_flags[] written by system_openai_moderation or system_llama_guard",
    facecolor=C_SUBFILL)

box(LM + bw3 + gap3, s3_y, bw3, s3_h,
    "Participant flag detection",
    "messages[].flagged == true  or  messages[].user_flag.category set",
    facecolor=C_SUBFILL)

# ════════════════════════════════════════════════════════════════════════════
# 4. Reviewer assignment engine
# ════════════════════════════════════════════════════════════════════════════
down_arrow(CX, s3_y, 5.68)
section_label(5.46, "4. Reviewer assignment engine")

s4_y, s4_h = 4.14, 1.18
gap4 = 0.14
bw4  = (W - 2 * gap4) / 3

bullet_box(
    LM, s4_y, bw4, s4_h,
    "LLM-flag assignment",
    "reason = 'llm_flag'  |  automatic",
    ["Change stream trigger",
     "Assigned to all reviewers",
     "Idempotent bulk write"])

bullet_box(
    LM + bw4 + gap4, s4_y, bw4, s4_h,
    "Participant-flag assignment",
    "reason = 'participant_flag'  |  automatic",
    ["Change stream trigger",
     "Assigned to all reviewers",
     "Idempotent bulk write"])

bullet_box(
    LM + 2 * (bw4 + gap4), s4_y, bw4, s4_h,
    "Random-sample assignment",
    "reason = 'random_sample'  |  manual",
    ["Admin API trigger",
     "20% per experiment/participant",
     "Individual or shared mode"])

italicnote(s4_y - 0.20,
    "Messages may carry multiple reasons simultaneously; "
    "reviewer view deduplicates by message index")

# ════════════════════════════════════════════════════════════════════════════
# 5. Real-time push
# ════════════════════════════════════════════════════════════════════════════
down_arrow(CX, s4_y - 0.36, 3.50)
section_label(3.28, "5. Real-time push  (Socket.io)")

s5_y, s5_h = 2.74, 0.42
box(LM, s5_y, W, s5_h,
    "Server emits  conversation:updated  to all connected clients",
    "{conversation_id,  operationType,  changedPaths,  hasAssignedMessages}",
    facecolor=C_FILL)

# ════════════════════════════════════════════════════════════════════════════
# 6. Review dashboard
# ════════════════════════════════════════════════════════════════════════════
down_arrow(CX, s5_y, 2.36)
section_label(2.14, "6. Review dashboard  (React 18  +  Express REST API)")

s6_y, s6_h = 0.78, 1.22
gap6 = 0.14
bw6a = W * 0.455
bw6b = W - bw6a - gap6

bullet_box(
    LM, s6_y, bw6a, s6_h,
    "Reviewer interface",
    "Paginated conversation list  |  message-level flag view",
    ["Filter by project / experiment / date range",
     "Flag modal: severity + category + comment",
     "Mark message reviewed  (auto-flag severity 0)",
     "Live refresh on  conversation:updated  event"])

bullet_box(
    LM + bw6a + gap6, s6_y, bw6b, s6_h,
    "Admin panel",
    "Management & oversight",
    ["All flags: reviewer + participant sources",
     "User management  (create / delete)",
     "Assignment overview by reviewer",
     "Trigger / reset random-sample assignment"])

# ── Deployment footnote ───────────────────────────────────────────────────────
ax.plot([LM, RM], [0.60, 0.60], color="#CCCCCC", lw=0.6)
italicnote(0.43,
    "Deployment: Docker Compose (local dev)  |  Helm -> Kubernetes, project namespace (production)  "
    "|  CI/CD: GitHub Actions -> GitLab image registry",
    fs=7.1)
italicnote(0.24,
    "Branch strategy: dev -> dev environment  |  main -> production  "
    "|  Auth: JWT (jsonwebtoken) + bcrypt password hashing",
    fs=7.1)

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out_pdf = "/Users/yun/Dev/hpms/diagrams/realtime_monitoring_pipeline.pdf"
out_png = "/Users/yun/Dev/hpms/diagrams/realtime_monitoring_pipeline.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor=C_BG, dpi=300)
fig.savefig(out_png, format="png", bbox_inches="tight", facecolor=C_BG, dpi=300)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
