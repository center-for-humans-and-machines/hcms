"""
Consolidated procedure figure.
(a) Regression testing procedure   — 0828_regtest_procedure.drawio.pdf
(b) Real-time monitoring procedure — 0828_real_time_procedure.drawio.pdf

Four columns for (a): Dataset | Configuration | Simulated Conversation | Evaluation
Three columns for (b):          Configuration | Real Conversation      | Evaluation

ACM TIST minimal style matching other diagrams in this project.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.unicode_minus": False,
    "figure.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_BG     = "#FFFFFF"
C_TEXT   = "#111111"
C_MUTED  = "#5A5A5A"
C_LINE   = "#2E2E2E"
C_BORDER = "#6E6E6E"
C_FILL   = "#F5F5F5"
C_SUB    = "#FAFAFA"
C_ACCENT = "#4C566A"

# Role colours (match original drawio hues, kept muted for print)
COL_AGENT = "#2563EB"   # blue   – LLM conversational agent
COL_SIM   = "#16A34A"   # green  – LLM simulating user
COL_AIE   = "#EA580C"   # orange – AI evaluator
COL_HUM   = "#7C3AED"   # purple – human evaluator
COL_PSY   = "#9D174D"   # rose   – psychiatrist
COL_INFRA = "#6B7280"   # gray   – infrastructure

FIG_W, FIG_H = 7.5, 9.8
fig = plt.figure(figsize=(FIG_W, FIG_H), facecolor=C_BG)
ax  = fig.add_axes([0, 0, 1, 1], facecolor=C_BG)
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

# ── Primitive helpers ─────────────────────────────────────────────────────────

def rbox(x, y, w, h, fc=C_FILL, ec=C_BORDER, lw=0.85, r=0.05, z=3):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle=f"round,pad={r}",
                                linewidth=lw, edgecolor=ec,
                                facecolor=fc, zorder=z))

def txt(x, y, s, fs=8.0, ha="center", va="center",
        w="normal", c=C_TEXT, italic=False, z=5):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs,
            fontweight=w, color=c,
            style="italic" if italic else "normal", zorder=z)

def col_hdr(cx, y, label, fs=8.3):
    txt(cx, y, label, fs=fs, w="bold", c=C_ACCENT)

def ibox(x, y, w, h, label, fs=7.0):
    """Plain item box."""
    rbox(x, y, w, h, fc=C_SUB, ec=C_BORDER, lw=0.65, r=0.025)
    txt(x + w/2, y + h/2, label, fs=fs)

# ── Icon primitives ───────────────────────────────────────────────────────────

def agent_icon(cx, cy, sz, color, dashed=False):
    """Small robot-style icon: round head + square body."""
    ls = (0, (3, 2)) if dashed else "solid"
    fc = C_BG
    # body
    bw, bh = sz * 0.55, sz * 0.44
    ax.add_patch(FancyBboxPatch((cx - bw/2, cy - sz*0.28), bw, bh,
                                boxstyle="round,pad=0.01",
                                linewidth=0.8, linestyle=ls,
                                edgecolor=color, facecolor=fc, zorder=6))
    # head
    hr = sz * 0.24
    ax.add_patch(plt.Circle((cx, cy + sz*0.26), hr,
                             linewidth=0.8, linestyle=ls,
                             edgecolor=color, facecolor=fc, zorder=6))
    # antenna dot
    ax.add_patch(plt.Circle((cx, cy + sz*0.26 + hr + sz*0.06),
                             sz * 0.07, color=color, zorder=7))


def person_icon(cx, cy, sz, color):
    """Simple person: circle head + oval torso."""
    ax.add_patch(plt.Circle((cx, cy + sz*0.30), sz*0.24,
                             linewidth=0.8, edgecolor=color,
                             facecolor=C_BG, zorder=6))
    ax.add_patch(Ellipse((cx, cy - sz*0.10), sz*0.50, sz*0.46,
                          linewidth=0.8, edgecolor=color,
                          facecolor=C_BG, zorder=6))


def cube_icon(cx, cy, sz, color):
    """3-D cube outline."""
    s = sz * 0.40
    off = s * 0.38
    def poly(pts):
        ax.add_patch(plt.Polygon(pts, closed=True, fill=False,
                                 edgecolor=color, linewidth=0.75, zorder=6))
    poly([(cx-s, cy-s), (cx+s, cy-s), (cx+s, cy+s), (cx-s, cy+s)])
    poly([(cx-s, cy+s), (cx-s+off, cy+s+off),
          (cx+s+off, cy+s+off), (cx+s, cy+s)])
    poly([(cx+s, cy-s), (cx+s+off, cy-s+off),
          (cx+s+off, cy+s+off), (cx+s, cy+s)])


def cylinder(cx, cy, cw, ch, label):
    """Dataset cylinder."""
    ew, eh = cw, cw * 0.30
    # sides
    ax.add_patch(plt.Polygon(
        [(cx-ew/2, cy-ch/2+eh/2), (cx-ew/2, cy+ch/2-eh/2),
         (cx+ew/2, cy+ch/2-eh/2), (cx+ew/2, cy-ch/2+eh/2)],
        closed=True, facecolor=C_SUB, edgecolor=C_SUB, lw=0, zorder=3))
    ax.plot([cx-ew/2]*2, [cy-ch/2+eh/2, cy+ch/2-eh/2],
            color=C_BORDER, lw=0.8, zorder=4)
    ax.plot([cx+ew/2]*2, [cy-ch/2+eh/2, cy+ch/2-eh/2],
            color=C_BORDER, lw=0.8, zorder=4)
    # bottom ellipse
    ax.add_patch(Ellipse((cx, cy-ch/2+eh/2), ew, eh,
                          lw=0.8, edgecolor=C_BORDER, facecolor=C_SUB, zorder=4))
    # top ellipse
    ax.add_patch(Ellipse((cx, cy+ch/2-eh/2), ew, eh,
                          lw=0.8, edgecolor=C_BORDER, facecolor=C_SUB, zorder=5))
    txt(cx, cy-ch/2-0.16, label, fs=7.0)

# ── Compound components ───────────────────────────────────────────────────────

def config_group(x, y, w, h, icon_fn, title, items):
    """
    Rounded group box: icon+title in left ~40%, stacked item boxes in right ~60%.
    """
    rbox(x, y, w, h, fc=C_FILL, ec=C_BORDER, lw=0.9, r=0.06)

    # left zone centre x
    lz = x + w * 0.22
    icon_fn(lz, y + h * 0.67, h * 0.30)
    txt(lz, y + h * 0.28, title, fs=6.3, w="bold")

    # right zone: stacked item boxes
    n  = len(items)
    ix = x + w * 0.42
    iw = w - w * 0.42 - 0.06
    pad_top = 0.09
    gap     = 0.055
    ih = (h - pad_top - gap * (n - 1)) / n - 0.02

    for k, label in enumerate(items):
        iy = y + h - pad_top - (k + 1) * ih - k * gap
        ibox(ix, iy, iw, ih, label, fs=6.8)


def eval_group(x, y, w, h, icon_fn, title, items):
    """Same layout as config_group, used for evaluation column."""
    config_group(x, y, w, h, icon_fn, title, items)


def conv_panel(x, y, w, h, left_color, right_color,
               left_dashed=True, right_dashed=False):
    """
    Rounded outer border containing alternating speech-bubble pairs.
    Left bubbles = simulated/participant side; right = agent side.
    """
    rbox(x, y, w, h, fc=C_SUB, ec=C_BORDER, lw=0.9, r=0.10)

    icon_w = 0.22              # fixed icon zone width (left or right)
    pad_x  = 0.06             # inner horizontal padding
    bw     = w - 2*icon_w - 2*pad_x   # bubble width fills remaining space
    bh     = h * 0.155        # bubble height
    gap    = h * 0.032        # vertical gap between bubbles
    pad_t  = h * 0.050        # top padding inside panel
    isz    = min(icon_w * 0.75, 0.22)  # icon size

    pairs = [
        ("left",  left_color,  left_dashed),
        ("right", right_color, right_dashed),
        ("left",  left_color,  left_dashed),
        ("right", right_color, right_dashed),
    ]

    for i, (side, color, dashed) in enumerate(pairs):
        by = y + h - pad_t - bh/2 - i * (bh + gap)
        ls = (0, (4, 2)) if dashed else "solid"

        if side == "left":
            icon_cx = x + pad_x + icon_w / 2
            bx_left = x + pad_x + icon_w + pad_x
        else:
            icon_cx = x + w - pad_x - icon_w / 2
            bx_left = x + w - pad_x - icon_w - pad_x - bw

        bcx = bx_left + bw / 2

        # bubble
        ax.add_patch(FancyBboxPatch((bx_left, by - bh/2), bw, bh,
                                    boxstyle="round,pad=0.04",
                                    linewidth=0.85, linestyle=ls,
                                    edgecolor=color, facecolor=C_BG, zorder=5))
        # content lines
        lp = bw * 0.10
        ax.plot([bx_left+lp, bx_left+bw-lp],      [by+bh*0.17]*2,
                color=color, lw=0.85, solid_capstyle="round", zorder=6)
        ax.plot([bx_left+lp, bx_left+bw-lp*1.7],  [by-bh*0.17]*2,
                color=color, lw=0.85, solid_capstyle="round", zorder=6)

        # icon
        if side == "left":
            agent_icon(icon_cx, by, isz, left_color,  dashed=left_dashed)
        else:
            agent_icon(icon_cx, by, isz, right_color, dashed=right_dashed)

    # ellipsis
    txt(x + w/2, y + h * 0.065, "...", fs=9, c=C_MUTED)


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (a) – Regression Testing
# ═══════════════════════════════════════════════════════════════════════════════

A_TOP = 9.62
A_BOT = 5.10

# Column x boundaries  (4 columns)
AX1, AX2 = 0.28, 1.22   # Dataset      w=0.94
AX3, AX4 = 1.30, 2.90   # Config       w=1.60
AX5, AX6 = 2.98, 4.62   # Conv         w=1.64
AX7, AX8 = 4.70, 7.22   # Eval         w=2.52

HDR_A = A_TOP - 0.34

# panel label
txt(0.28, A_TOP - 0.10, "(a)  Regression Testing",
    fs=9.0, ha="left", w="bold")

col_hdr((AX1+AX2)/2,  HDR_A, "Dataset")
col_hdr((AX3+AX4)/2,  HDR_A, "Configuration")
col_hdr((AX5+AX6)/2,  HDR_A, "Simulated Conversation")
col_hdr((AX7+AX8)/2,  HDR_A, "Evaluation")

# ── Dataset ───────────────────────────────────────────────────────────────────
CYL_W = AX2 - AX1 - 0.08
CYL_H = 0.64
ds_cx  = (AX1 + AX2) / 2
cylinder(ds_cx, A_TOP - 1.00, CYL_W, CYL_H, "Standardized\nPrompts")
cylinder(ds_cx, A_TOP - 2.58, CYL_W, CYL_H, "Open-Ended\nPrompts")

# ── Configuration (3 groups) ──────────────────────────────────────────────────
CF_W  = AX4 - AX3        # 1.60
G_H   = 1.06
G_GAP = 0.11

g1y = A_TOP - 0.42 - G_H         # top group bottom-y
g2y = g1y - G_GAP - G_H
g3y = g2y - G_GAP - G_H

config_group(AX3, g1y, CF_W, G_H,
             lambda cx, cy, s: agent_icon(cx, cy, s, COL_AGENT),
             "LLM Conv.\nAgent",
             ["API endpoint", "System Prompt", "Temperature"])

config_group(AX3, g2y, CF_W, G_H,
             lambda cx, cy, s: agent_icon(cx, cy, s, COL_SIM, dashed=True),
             "LLM Sim.\nUser",
             ["API endpoint", "System Prompt", "Temperature"])

config_group(AX3, g3y, CF_W, G_H,
             lambda cx, cy, s: cube_icon(cx, cy, s, COL_INFRA),
             "Infrastructure",
             ["GitHub Action", "MongoDB", "Langfuse"])

# ── Simulated Conversation ────────────────────────────────────────────────────
CV_W = AX6 - AX5
CV_Y = g3y
CV_H = g1y + G_H - g3y
conv_panel(AX5, CV_Y, CV_W, CV_H,
           left_color=COL_SIM, right_color=COL_AGENT,
           left_dashed=True, right_dashed=False)

# ── Evaluation (2 groups, spanning same height as config) ─────────────────────
EV_W   = AX8 - AX7
EV_GAP = 0.11
EV_TOT = CV_H                   # same total height as conv panel
# AI Evaluator (2 items) gets ~40%, Human (3 items) gets ~60%
EV_H1 = (EV_TOT - EV_GAP) * 0.42
EV_H2 = EV_TOT - EV_H1 - EV_GAP

ev2y = g3y                      # Human Evaluator at bottom
ev1y = ev2y + EV_H2 + EV_GAP   # AI Evaluator above it

eval_group(AX7, ev1y, EV_W, EV_H1,
           lambda cx, cy, s: agent_icon(cx, cy, s, COL_AIE),
           "AI Evaluator",
           ["Moderation API", "Llama Guard"])

eval_group(AX7, ev2y, EV_W, EV_H2,
           lambda cx, cy, s: person_icon(cx, cy, s, COL_HUM),
           "Human\nEvaluator",
           ["Nuanced failure", "Longer context", "Sycophancy"])

# ── Divider line ──────────────────────────────────────────────────────────────
ax.plot([0.20, 7.30], [4.96, 4.96], color=C_LINE, lw=0.75)

# ═══════════════════════════════════════════════════════════════════════════════
# PANEL (b) – Real-Time Monitoring
# ═══════════════════════════════════════════════════════════════════════════════

B_TOP = 4.86
B_BOT = 0.28

# Column x boundaries (3 columns; no Dataset)
BX1, BX2 = 0.28, 2.38   # Config   w=2.10   (wider – spans former dataset+config)
BX3, BX4 = 2.46, 4.62   # Conv     w=2.16
BX5, BX6 = 4.70, 7.22   # Eval     w=2.52   (x-aligned with panel a eval)

HDR_B = B_TOP - 0.34

txt(0.28, B_TOP - 0.10, "(b)  Real-Time Monitoring",
    fs=9.0, ha="left", w="bold")

col_hdr((BX1+BX2)/2, HDR_B, "Configuration")
col_hdr((BX3+BX4)/2, HDR_B, "Real Conversation")
col_hdr((BX5+BX6)/2, HDR_B, "Evaluation")

# ── Configuration (2 groups) ──────────────────────────────────────────────────
BCF_W  = BX2 - BX1       # 2.10
BG_GAP = 0.14
BG_TOT = B_TOP - 0.42 - B_BOT   # usable height
BG_H   = (BG_TOT - BG_GAP) / 2  # each group

bg1y = B_TOP - 0.42 - BG_H
bg2y = bg1y - BG_GAP - BG_H

config_group(BX1, bg1y, BCF_W, BG_H,
             lambda cx, cy, s: agent_icon(cx, cy, s, COL_AGENT),
             "LLM Conv.\nAgent",
             ["API endpoint", "System Prompt", "Temperature"])

config_group(BX1, bg2y, BCF_W, BG_H,
             lambda cx, cy, s: cube_icon(cx, cy, s, COL_INFRA),
             "Infrastructure",
             ["GitHub Action", "MongoDB", "Langfuse"])

# ── Real Conversation ─────────────────────────────────────────────────────────
BCV_W = BX4 - BX3
BCV_Y = bg2y
BCV_H = bg1y + BG_H - bg2y
# Real-time: participant on left (person, solid), agent on right (robot, solid)
conv_panel(BX3, BCV_Y, BCV_W, BCV_H,
           left_color=COL_SIM, right_color=COL_AGENT,
           left_dashed=False, right_dashed=False)

# Replace left agent icons with person icon for real participant
# (done inline: left_dashed=False already gives solid green bubbles)

# ── Evaluation (2 groups) ─────────────────────────────────────────────────────
BEV_W   = BX6 - BX5
BEV_GAP = 0.14
BEV_TOT = BCV_H
BEV_H1  = (BEV_TOT - BEV_GAP) * 0.42
BEV_H2  = BEV_TOT - BEV_H1 - BEV_GAP

bev2y = bg2y
bev1y = bev2y + BEV_H2 + BEV_GAP

eval_group(BX5, bev1y, BEV_W, BEV_H1,
           lambda cx, cy, s: agent_icon(cx, cy, s, COL_AIE),
           "AI Evaluator",
           ["Moderation API", "Llama Guard"])

eval_group(BX5, bev2y, BEV_W, BEV_H2,
           lambda cx, cy, s: person_icon(cx, cy, s, COL_PSY),
           "Psychiatrist",
           ["Nuanced failure", "Longer context", "Sycophancy"])

# ── Footer ────────────────────────────────────────────────────────────────────
ax.plot([0.20, 7.30], [0.22, 0.22], color="#DDDDDD", lw=0.5)
txt(FIG_W/2, 0.13,
    "Dashed outlines = LLM simulating user;  "
    "AI Evaluator = OpenAI Moderation API + LlamaGuard-4-12B;  "
    "Psychiatrist = domain expert human reviewer",
    fs=6.5, c=C_MUTED, italic=True)

# ═══════════════════════════════════════════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════════════════════════════════════════
import matplotlib.pyplot as _plt
_plt.rcParams["path.simplify"] = False   # keep all detail

out_pdf = "/Users/yun/Dev/hpms/notebooks/procedure_diagram.pdf"
out_png = "/Users/yun/Dev/hpms/notebooks/procedure_diagram.png"
fig.savefig(out_pdf, format="pdf", bbox_inches="tight", facecolor=C_BG, dpi=300)
fig.savefig(out_png, format="png", bbox_inches="tight", facecolor=C_BG, dpi=300)
print(f"Saved:\n  {out_pdf}\n  {out_png}")
