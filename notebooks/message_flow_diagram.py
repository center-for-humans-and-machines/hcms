"""
Sequence / swim-lane flowchart: how user messages (u) and conversational agent
messages (c) travel through the system.

FIG_W = ACM TIST \textwidth (7.0 in) so matplotlib font sizes == rendered pt sizes.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9.0,
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
C_ACCENT  = "#4C566A"
C_ARROW   = "#2E2E2E"

# Font sizes — equal to rendered pt sizes when figure included at \textwidth
FS_HEADER  = 10.0
FS_MSG     = 10.5
FS_DESC    =  8.5
FS_PHASE   =  9.5
FS_DIAMOND =  9.0
FS_TITLE   = 12.0
FS_BRANCH  =  8.0
FS_SMALL   =  7.5

# ── Canvas ────────────────────────────────────────────────────────────────────
FIG_W, FIG_H = 7.0, 7.4
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Column layout ─────────────────────────────────────────────────────────────
LM = 0.20
RM = 6.80
n  = 5
lane_w = (RM - LM) / n   # 1.32 in per lane

xF = LM + 0.5 * lane_w   # 0.86  Frontend
xB = LM + 1.5 * lane_w   # 2.18  Backend
xM = LM + 2.5 * lane_w   # 3.50  MongoDB
xL = LM + 3.5 * lane_w   # 4.82  LLM
xD = LM + 4.5 * lane_w   # 6.14  Monitoring Dashboard

HDR_H   = 0.50
HDR_BOT = FIG_H - 0.70 - HDR_H   # = 6.20

# ── Row layout (computed bottom-up for guaranteed clearance) ─────────────────
# ROW=0.50: at FS_MSG=10.5pt (h≈0.146in) and FS_DESC=8.5pt (h≈0.118in) with
# offsets ±0.075, clearance between adjacent rows ≈ 0.086 in.
ROW = 0.50

FOOT_LINE = 0.28   # y of footnote separator
FOOT_TEXT = 0.15   # y of footnote text
LL_BOT    = 0.36   # lifelines end just above footnote area

# Work bottom-up so nothing overflows
y_c5  = LL_BOT + 0.20          # 0.56  stream c to participant
y_dia = y_c5  + 0.42           # 0.98  decision diamond
y_c4  = y_dia + 0.40           # 1.38  safety check
y_c3  = y_c4  + ROW            # 1.88  change stream auto-assign
y_c2  = y_c3  + ROW            # 2.38  persist c + flags
y_c1  = y_c2  + ROW            # 2.88  response generated
y_div = y_c1  + 0.24           # 3.12  phase divider
y_u5  = y_div + 0.26           # 3.38  u: confirmed
y_u4  = y_u5  + ROW            # 3.88
y_u3  = y_u4  + ROW            # 4.38
y_u2  = y_u3  + ROW            # 4.88
y_u1  = y_u2  + ROW            # 5.38
y_u0  = y_u1  + ROW            # 5.88
y_u   = [y_u0, y_u1, y_u2, y_u3, y_u4, y_u5]

ACT_W = 0.10

# ── Helpers ───────────────────────────────────────────────────────────────────

def header(x, label):
    w = lane_w - 0.10
    ax.add_patch(Rectangle((x - w/2, HDR_BOT), w, HDR_H,
                            linewidth=0.9, edgecolor=C_BORDER,
                            facecolor=C_FILL, zorder=3))
    ax.text(x, HDR_BOT + HDR_H/2, label,
            ha="center", va="center",
            fontsize=FS_HEADER, fontweight="bold", color=C_TEXT,
            zorder=4, linespacing=1.2)


def lifeline(x):
    ax.plot([x, x], [HDR_BOT, LL_BOT],
            color=C_BORDER, lw=0.6,
            linestyle=(0, (4, 3)), zorder=1)


def activation_bar(y_top, y_bot):
    ax.add_patch(Rectangle((xB - ACT_W/2, y_bot), ACT_W, y_top - y_bot,
                            linewidth=0.7, edgecolor=C_BORDER,
                            facecolor=C_BG, zorder=3))


def arrow(x1, x2, y, msg, desc=None):
    """Horizontal arrow: bold msg above, italic desc below."""
    ax.plot([x1, x2], [y, y], color=C_ARROW, lw=0.85, zorder=4,
            solid_capstyle="butt")
    ax.annotate("", xy=(x2, y), xytext=(x2 - 0.001*(1 if x2>x1 else -1), y),
                arrowprops=dict(arrowstyle="-|>", lw=0.85,
                                color=C_ARROW, mutation_scale=8), zorder=5)
    mx = (x1 + x2) / 2
    ax.text(mx, y + 0.075, msg,
            ha="center", va="bottom",
            fontsize=FS_MSG, fontweight="bold", color=C_TEXT, zorder=6)
    if desc:
        ax.text(mx, y - 0.065, desc,
                ha="center", va="top",
                fontsize=FS_DESC, color=C_MUTED, style="italic", zorder=6)


def diamond(x, y, label):
    hw, hh = 0.22, 0.14
    pts = [(x, y+hh), (x+hw, y), (x, y-hh), (x-hw, y)]
    ax.add_patch(Polygon(pts, closed=True, linewidth=0.9,
                         edgecolor=C_BORDER, facecolor=C_BG, zorder=5))
    ax.text(x, y + hh + 0.055, label,
            ha="center", va="bottom",
            fontsize=FS_DIAMOND, fontweight="bold", color=C_ACCENT, zorder=6)


def phase_label(y, text):
    ax.text(LM + 0.08, y + 0.045, text,
            ha="left", va="bottom",
            fontsize=FS_PHASE, fontweight="bold", color=C_ACCENT, zorder=4)


def phase_divider_line(y):
    ax.plot([LM+0.04, RM-0.04], [y, y],
            color="#CCCCCC", lw=0.5, linestyle=(0, (5, 4)), zorder=1)


# ════════════════════════════════════════════════════════════════════════════
# DRAW
# ════════════════════════════════════════════════════════════════════════════

# Title
ax.text(FIG_W/2, FIG_H - 0.26,
        "Message Flow: User (u) and Conversational Agent (c) Turns",
        ha="center", va="center",
        fontsize=FS_TITLE, fontweight="bold", color=C_TEXT)
ax.plot([LM, RM], [FIG_H - 0.48, FIG_H - 0.48],
        color=C_LINE, lw=0.7, solid_capstyle="butt")

# Headers & lifelines
for x, lbl in [(xF, "Frontend"),
               (xB, "Backend"),
               (xM, "MongoDB"),
               (xL, "LLM"),
               (xD, "Monitoring\nDashboard")]:
    header(x, lbl)
    lifeline(x)

activation_bar(y_top=y_u[0] + 0.16, y_bot=y_c5 - 0.14)

# ── u phase ──────────────────────────────────────────────────────────────────
phase_label(y_u[0] + 0.10, "User message  (u)")

arrow(xF, xB, y_u[0], "u", "sends message")
arrow(xB, xM, y_u[1], "u", "write to Conversations")
arrow(xM, xD, y_u[2], "u", "change stream → Socket.io notify")
arrow(xB, xL, y_u[3], "u", "forward conversation history")
arrow(xL, xB, y_u[4], "u", "begin generating response")
arrow(xB, xF, y_u[5], "u", "confirmed")

# ── phase divider ─────────────────────────────────────────────────────────────
phase_divider_line(y_div)

# ── c phase ──────────────────────────────────────────────────────────────────
phase_label(y_c1 + 0.10, "Conversational agent response  (c)")

arrow(xL, xB, y_c1, "c", "response generated")
arrow(xB, xM, y_c2, "c", "persist c + reviewer_flags")
arrow(xM, xD, y_c3, "c", "change stream → auto-assign reviewers")
arrow(xB, xL, y_c4, "c", "LlamaGuard / OpenAI Moderation check")

# ── decision diamond ─────────────────────────────────────────────────────────
diamond(xB, y_dia, "Is C safe?")

# ── final c ──────────────────────────────────────────────────────────────────
arrow(xB, xF, y_c5, "c", "stream c to participant")

# ── Vertical connector: diamond → final arrow ─────────────────────────────────
ax.plot([xB, xB], [y_dia - 0.14, y_c5 + 0.01],
        color=C_ARROW, lw=0.75, linestyle=(0, (3, 2)), zorder=3)

# ── yes / no branch labels ────────────────────────────────────────────────────
ax.text(xB + 0.11, (y_dia + y_c5) / 2, "yes",
        ha="left", va="center",
        fontsize=FS_BRANCH, color=C_MUTED, style="italic")

no_stub_x = xM - 0.16
ax.plot([xB + 0.22, no_stub_x], [y_dia, y_dia],
        color=C_MUTED, lw=0.75, zorder=4)
ax.annotate("", xy=(no_stub_x, y_dia), xytext=(no_stub_x - 0.01, y_dia),
            arrowprops=dict(arrowstyle="-|>", lw=0.75,
                            color=C_MUTED, mutation_scale=6), zorder=4)
ax.text((xB + 0.22 + no_stub_x)/2, y_dia + 0.075, "no → flag for review",
        ha="center", va="bottom",
        fontsize=FS_BRANCH, color=C_MUTED, style="italic")

# ── loop bracket ─────────────────────────────────────────────────────────────
bx    = RM + 0.02
b_top = y_u[0] + 0.16
b_bot = y_c5 - 0.12
tick  = 0.08
ax.plot([bx, bx+tick, bx+tick, bx], [b_top, b_top, b_bot, b_bot],
        color=C_BORDER, lw=0.65, zorder=2)
ax.text(bx + tick + 0.05, (b_top + b_bot)/2,
        "repeats\nper turn",
        ha="left", va="center",
        fontsize=FS_SMALL, color=C_MUTED, style="italic")

# ── footnote (single line, safely below activation bar) ───────────────────────
ax.plot([LM, RM], [FOOT_LINE, FOOT_LINE], color="#DDDDDD", lw=0.5)
ax.text(FIG_W/2, FOOT_TEXT,
        "Monitoring Dashboard: hpms-dashboard (Express + Socket.io + React 18)"
        "  |  reviewer_flags: LlamaGuard-4-12B / omni-moderation-2024-09-26",
        ha="center", va="center",
        fontsize=FS_SMALL, color=C_MUTED, style="italic")

# ════════════════════════════════════════════════════════════════════════════
# SAVE
# ════════════════════════════════════════════════════════════════════════════
out_pdf = "/Users/yun/Dev/hpms/message_flow_diagram.pdf"
out_png = "/Users/yun/Dev/hpms/message_flow_diagram.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor=C_BG, dpi=300)
fig.savefig(out_png, format="png", bbox_inches="tight", facecolor=C_BG, dpi=300)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
