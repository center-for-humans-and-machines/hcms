"""
Sequence / swim-lane flowchart: how user messages (u) and companion
messages (c) travel through the system.

Based on dashboard-flowchart.pdf and hpms-dashboard repository.
Style matches other ACM TIST-style diagrams in this project.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_BG      = "#FFFFFF"
C_TEXT    = "#111111"
C_MUTED   = "#5A5A5A"
C_LINE    = "#2E2E2E"
C_BORDER  = "#6E6E6E"
C_FILL    = "#F5F5F5"
C_SUBFILL = "#FAFAFA"
C_ACCENT  = "#4C566A"
C_ARROW   = "#2E2E2E"

# ── Canvas (landscape) ────────────────────────────────────────────────────────
FIG_W, FIG_H = 10.0, 7.5
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Column layout ─────────────────────────────────────────────────────────────
LM = 0.30
RM = 9.70
n  = 5
lane_w = (RM - LM) / n   # 1.88

xF = LM + 0.5 * lane_w   # 1.24  Frontend
xB = LM + 1.5 * lane_w   # 3.12  Backend
xM = LM + 2.5 * lane_w   # 5.00  MongoDB
xL = LM + 3.5 * lane_w   # 6.88  LLM
xD = LM + 4.5 * lane_w   # 8.76  Monitoring Dashboard

HDR_H   = 0.52
HDR_BOT = 7.00 - HDR_H   # 6.48  bottom edge of header boxes
LL_BOT  = 0.30            # lifelines end here

# Backend activation bar (thin solid rect on top of dashed lifeline)
ACT_W = 0.11

# ── Row y-coordinates (arrows, top → bottom) ──────────────────────────────────
# u phase
y_u = [6.08, 5.58, 5.08, 4.58, 4.08, 3.58]

# phase divider
y_div = 3.24

# c phase
y_c1 = 2.90   # LLM → Backend        "response c generated"
y_c2 = 2.40   # Backend → MongoDB    "persist c + reviewer_flags"
y_c3 = 1.90   # Backend → Monitoring "change stream → auto-assign"
y_c4 = 1.40   # Backend → LLM        "safety check (LlamaGuard / OpenAI Moderation)"

y_dia = 0.96  # diamond               "Is C safe?"

y_c5 = 0.52   # Backend → Frontend   "stream c to participant"

# ── Helpers ───────────────────────────────────────────────────────────────────

def header(x, label):
    """Swim-lane column header."""
    w = lane_w - 0.10
    ax.add_patch(Rectangle((x - w/2, HDR_BOT), w, HDR_H,
                            linewidth=0.9, edgecolor=C_BORDER,
                            facecolor=C_FILL, zorder=3))
    ax.text(x, HDR_BOT + HDR_H/2, label,
            ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=C_TEXT, zorder=4)


def lifeline(x):
    """Dashed vertical lifeline."""
    ax.plot([x, x], [HDR_BOT, LL_BOT],
            color=C_BORDER, lw=0.65,
            linestyle=(0, (4, 3)), zorder=1)


def activation_bar(y_top, y_bot):
    """Backend activation bar (UML-style)."""
    ax.add_patch(Rectangle((xB - ACT_W/2, y_bot), ACT_W, y_top - y_bot,
                            linewidth=0.7, edgecolor=C_BORDER,
                            facecolor=C_BG, zorder=3))


def arrow(x1, x2, y, msg, desc=None):
    """
    Horizontal arrow from x1 to x2 at height y.
    msg  = 'u' or 'c'  (bold, above midpoint)
    desc = short italic description (below midpoint)
    """
    # Horizontal line
    ax.plot([x1, x2], [y, y], color=C_ARROW, lw=0.85, zorder=4,
            solid_capstyle="butt")
    # Arrowhead (pointing toward x2)
    ax.annotate("", xy=(x2, y), xytext=(x2 - 0.001 * (1 if x2 > x1 else -1), y),
                arrowprops=dict(arrowstyle="-|>", lw=0.85,
                                color=C_ARROW, mutation_scale=7.5),
                zorder=5)

    mx = (x1 + x2) / 2
    # Bold message label above
    ax.text(mx, y + 0.105, msg,
            ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=C_TEXT, zorder=6)
    # Italic description below
    if desc:
        ax.text(mx, y - 0.095, desc,
                ha="center", va="top",
                fontsize=6.7, color=C_MUTED, style="italic", zorder=6)


def diamond(x, y, label):
    """Decision diamond with label below."""
    hw, hh = 0.22, 0.15
    pts = [(x, y + hh), (x + hw, y), (x, y - hh), (x - hw, y)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=0.9,
                         edgecolor=C_BORDER, facecolor=C_BG, zorder=5))
    ax.text(x, y + hh + 0.07, label,
            ha="center", va="bottom",
            fontsize=7.8, fontweight="bold", color=C_ACCENT, zorder=6)


def phase_label(y, text):
    """Bold phase heading just above a row."""
    ax.text(LM + 0.12, y + 0.06, text,
            ha="left", va="bottom",
            fontsize=7.8, fontweight="bold",
            color=C_ACCENT, zorder=4)


def phase_divider_line(y):
    ax.plot([LM + 0.05, RM - 0.05], [y, y],
            color="#CCCCCC", lw=0.55, linestyle=(0, (5, 4)), zorder=1)


# ════════════════════════════════════════════════════════════════════════════
# DRAW
# ════════════════════════════════════════════════════════════════════════════

# Title
ax.text(FIG_W / 2, 7.34,
        "Message Flow: User (u) and Companion (c) Turns",
        ha="center", va="center",
        fontsize=12, fontweight="bold", color=C_TEXT)
ax.plot([LM, RM], [7.10, 7.10], color=C_LINE, lw=0.7, solid_capstyle="butt")

# Headers & lifelines
for x, lbl in [(xF, "Frontend"),
               (xB, "Backend"),
               (xM, "MongoDB"),
               (xL, "LLM"),
               (xD, "Monitoring\nDashboard")]:
    header(x, lbl)
    lifeline(x)

# Activation bar on Backend (spans full sequence)
activation_bar(y_top=y_u[0] + 0.18, y_bot=y_c5 - 0.16)

# ── u phase ──────────────────────────────────────────────────────────────────
phase_label(y_u[0] + 0.22, "User message  (u)")

arrow(xF, xB, y_u[0], "u", "participant sends message")
arrow(xB, xM, y_u[1], "u", "write to Conversations")
arrow(xM, xD, y_u[2], "u", "change stream -> Socket.io notify")
arrow(xB, xL, y_u[3], "u", "forward conversation history")
arrow(xL, xB, y_u[4], "u", "begin generating response")
arrow(xB, xF, y_u[5], "u", "message received, confirmed")

# ── phase divider ─────────────────────────────────────────────────────────────
phase_divider_line(y_div)

# ── c phase ──────────────────────────────────────────────────────────────────
phase_label(y_c1 + 0.22, "Companion response  (c)")

arrow(xL, xB, y_c1, "c", "response c generated")
arrow(xB, xM, y_c2, "c", "persist c + reviewer_flags")
arrow(xM, xD, y_c3, "c", "change stream -> auto-assign reviewers")
arrow(xB, xL, y_c4, "c", "safety check  (LlamaGuard / OpenAI Moderation API)")

# ── decision diamond ─────────────────────────────────────────────────────────
diamond(xB, y_dia, "Is C safe?")

# ── final c ──────────────────────────────────────────────────────────────────
arrow(xB, xF, y_c5, "c", "stream c to participant")

# ── Vertical connector from diamond to final arrow (dashed) ──────────────────
ax.plot([xB, xB], [y_dia - 0.15, y_c5 + 0.01],
        color=C_ARROW, lw=0.75, linestyle=(0, (3, 2)), zorder=3)

# ── "safe" / "unsafe" branch labels ─────────────────────────────────────────
# "yes" label along the downward connector
ax.text(xB + 0.12, (y_dia + y_c5) / 2, "yes",
        ha="left", va="center",
        fontsize=7.0, color=C_MUTED, style="italic")

# "no" branch: stub rightward into MongoDB column, with label above
no_stub_x = xM - 0.15
ax.plot([xB + 0.22, no_stub_x], [y_dia, y_dia],
        color=C_MUTED, lw=0.75, zorder=4)
ax.annotate("", xy=(no_stub_x, y_dia), xytext=(no_stub_x - 0.01, y_dia),
            arrowprops=dict(arrowstyle="-|>", lw=0.75,
                            color=C_MUTED, mutation_scale=6),
            zorder=4)
ax.text((xB + 0.22 + no_stub_x) / 2, y_dia + 0.09, "no  ->  flag for review",
        ha="center", va="bottom",
        fontsize=6.8, color=C_MUTED, style="italic")

# ── loop bracket on right margin ─────────────────────────────────────────────
bx = RM + 0.02
b_top = y_u[0] + 0.18
b_bot = y_c5 - 0.14
tick = 0.08
ax.plot([bx, bx + tick, bx + tick, bx],
        [b_top, b_top, b_bot, b_bot],
        color=C_BORDER, lw=0.7, zorder=2)
ax.text(bx + tick + 0.06, (b_top + b_bot) / 2,
        "repeats\nper turn",
        ha="left", va="center",
        fontsize=6.5, color=C_MUTED, style="italic")

# ── footnote ─────────────────────────────────────────────────────────────────
ax.plot([LM, RM], [0.24, 0.24], color="#DDDDDD", lw=0.5)
ax.text(FIG_W / 2, 0.14,
        "Monitoring Dashboard = hpms-dashboard (Express + Socket.io + React 18)  "
        "|  reviewer_flags written by hpms monitoring (LlamaGuard-4-12B / omni-moderation-2024-09-26)",
        ha="center", va="center",
        fontsize=6.5, color=C_MUTED, style="italic")

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out_pdf = "/Users/yun/Dev/hpms/message_flow_diagram.pdf"
out_png = "/Users/yun/Dev/hpms/message_flow_diagram.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor=C_BG, dpi=300)
fig.savefig(out_png, format="png", bbox_inches="tight", facecolor=C_BG, dpi=300)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
