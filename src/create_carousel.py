import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as mpatches
import os

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
PDF_PATH = os.path.join(OUTPUT_DIR, "linkedin_carousel.pdf")

DPI = 108
SIZE_IN = 1080 / DPI

# ── Palette ──────────────────────────────────────────────────────────
BG_DARK    = "#0F1B2D"
BG_CARD    = "#162238"
ACCENT     = "#00B4D8"
ACCENT2    = "#0077B6"
WHITE      = "#FFFFFF"
LIGHT_GRAY = "#B0BEC5"
ORANGE     = "#FF8C00"
GREEN      = "#00C853"
RED_SOFT   = "#EF5350"
MUTED_BLUE = "#4FC3F7"


def new_slide():
    fig = plt.figure(figsize=(SIZE_IN, SIZE_IN), dpi=DPI, facecolor=BG_DARK)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(BG_DARK)
    ax.axis("off")
    return fig, ax


def add_accent_bar(ax, y=0.92, width=0.12):
    ax.plot([0.5 - width / 2, 0.5 + width / 2], [y, y],
            color=ACCENT, lw=4, solid_capstyle="round", transform=ax.transAxes)


def add_slide_number(ax, num, total=5):
    ax.text(0.95, 0.04, f"{num}/{total}",
            color=LIGHT_GRAY, fontsize=11, ha="right", va="bottom",
            fontfamily="sans-serif", alpha=0.5, transform=ax.transAxes)


def add_branding(ax):
    ax.text(0.05, 0.04, "Customer Churn Prediction",
            color=LIGHT_GRAY, fontsize=10, ha="left", va="bottom",
            fontfamily="sans-serif", alpha=0.4, transform=ax.transAxes)


# ═════════════════════════════════════════════════════════════════════
#  SLIDE 1 — Hook
# ═════════════════════════════════════════════════════════════════════
def slide_1():
    fig, ax = new_slide()

    # decorative circle
    circle = plt.Circle((0.82, 0.78), 0.18, color=ACCENT, alpha=0.07, transform=ax.transAxes)
    ax.add_patch(circle)
    circle2 = plt.Circle((0.15, 0.25), 0.12, color=ACCENT2, alpha=0.06, transform=ax.transAxes)
    ax.add_patch(circle2)

    # "BUSINESS ANALYTICS PROJECT" tag
    tag_rect = mpatches.FancyBboxPatch(
        (0.24, 0.72), 0.52, 0.06, boxstyle="round,pad=0.015",
        facecolor=ACCENT, edgecolor="none", alpha=0.15, transform=ax.transAxes)
    ax.add_patch(tag_rect)
    ax.text(0.5, 0.75, "BUSINESS ANALYTICS PROJECT",
            color=ACCENT, fontsize=13, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    ax.text(0.5, 0.62, "I built an ML model\nthat predicts customer\nchurn with 87% AUC",
            color=WHITE, fontsize=28, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif", linespacing=1.35)

    add_accent_bar(ax, y=0.44)

    ax.text(0.5, 0.37, "Here's what I found —\nand how I built it in Python",
            color=LIGHT_GRAY, fontsize=16, ha="center", va="center",
            fontfamily="sans-serif", linespacing=1.4)

    ax.text(0.5, 0.22, "Swipe →",
            color=ACCENT, fontsize=15, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    add_slide_number(ax, 1)
    add_branding(ax)
    return fig


# ═════════════════════════════════════════════════════════════════════
#  SLIDE 2 — The Problem
# ═════════════════════════════════════════════════════════════════════
def slide_2():
    fig, ax = new_slide()

    ax.text(0.5, 0.88, "The Business Problem",
            color=WHITE, fontsize=26, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")
    add_accent_bar(ax, y=0.82)

    # stat callout
    stat_rect = mpatches.FancyBboxPatch(
        (0.15, 0.62), 0.70, 0.14, boxstyle="round,pad=0.025",
        facecolor=ACCENT, edgecolor="none", alpha=0.12, transform=ax.transAxes)
    ax.add_patch(stat_rect)
    ax.text(0.5, 0.695, "26.5%  of customers churned",
            color=ACCENT, fontsize=22, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    body = (
        "The goal: predict who leaves before they do,\n"
        "so the retention team can intervene."
    )
    ax.text(0.5, 0.52, body,
            color=LIGHT_GRAY, fontsize=15, ha="center", va="center",
            fontfamily="sans-serif", linespacing=1.5)

    # data card
    card_rect = mpatches.FancyBboxPatch(
        (0.12, 0.22), 0.76, 0.18, boxstyle="round,pad=0.025",
        facecolor=BG_CARD, edgecolor=ACCENT2, linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(card_rect)
    ax.text(0.5, 0.365, "Dataset",
            color=ACCENT, fontsize=13, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")
    ax.text(0.5, 0.30, "IBM Telco  •  7,043 customers  •  20 features",
            color=WHITE, fontsize=14, ha="center", va="center",
            fontfamily="sans-serif")

    add_slide_number(ax, 2)
    add_branding(ax)
    return fig


# ═════════════════════════════════════════════════════════════════════
#  SLIDE 3 — Key EDA Finding
# ═════════════════════════════════════════════════════════════════════
def slide_3():
    fig, ax = new_slide()

    ax.text(0.5, 0.93, "The Biggest Churn Driver?  Contract Type.",
            color=WHITE, fontsize=19, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    chart_path = os.path.join(OUTPUT_DIR, "churn_by_contract_dark.png")
    chart_img = plt.imread(chart_path)
    img_ax = fig.add_axes([0.08, 0.22, 0.84, 0.67])
    img_ax.imshow(chart_img)
    img_ax.axis("off")

    ax.text(0.5, 0.12, "Locking customers into longer contracts\n"
                        "is the single biggest retention lever.",
            color=LIGHT_GRAY, fontsize=14, ha="center", va="center",
            fontfamily="sans-serif", linespacing=1.5,
            fontstyle="italic")

    add_slide_number(ax, 3)
    add_branding(ax)
    return fig


# ═════════════════════════════════════════════════════════════════════
#  SLIDE 4 — Model Results
# ═════════════════════════════════════════════════════════════════════
def slide_4():
    fig, ax = new_slide()

    ax.text(0.5, 0.90, "Model Comparison",
            color=WHITE, fontsize=26, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")
    add_accent_bar(ax, y=0.84)

    # Table rows
    models = [
        ("Logistic Regression", "0.83", LIGHT_GRAY, ""),
        ("Random Forest",       "0.85", LIGHT_GRAY, ""),
        ("XGBoost",             "0.87", ACCENT,     "← Best"),
    ]

    table_top = 0.72
    row_h = 0.09

    # header
    header_rect = mpatches.FancyBboxPatch(
        (0.10, table_top), 0.80, 0.06,
        boxstyle="round,pad=0.012",
        facecolor=ACCENT2, edgecolor="none", alpha=0.3, transform=ax.transAxes)
    ax.add_patch(header_rect)
    ax.text(0.18, table_top + 0.03, "Model",
            color=ACCENT, fontsize=13, ha="left", va="center",
            fontweight="bold", fontfamily="sans-serif")
    ax.text(0.70, table_top + 0.03, "AUC",
            color=ACCENT, fontsize=13, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    for i, (name, auc, color, tag) in enumerate(models):
        y = table_top - (i + 1) * row_h
        # row bg
        row_alpha = 0.08 if i % 2 == 0 else 0.04
        if name == "XGBoost":
            row_rect = mpatches.FancyBboxPatch(
                (0.10, y - 0.005), 0.80, 0.065,
                boxstyle="round,pad=0.012",
                facecolor=ACCENT, edgecolor=ACCENT, linewidth=1.5,
                alpha=0.15, transform=ax.transAxes)
            ax.add_patch(row_rect)
        else:
            row_rect = mpatches.FancyBboxPatch(
                (0.10, y - 0.005), 0.80, 0.065,
                boxstyle="round,pad=0.012",
                facecolor=WHITE, edgecolor="none",
                alpha=row_alpha, transform=ax.transAxes)
            ax.add_patch(row_rect)

        name_color = ACCENT if name == "XGBoost" else WHITE
        ax.text(0.18, y + 0.03, name,
                color=name_color, fontsize=14, ha="left", va="center",
                fontweight="bold" if name == "XGBoost" else "normal",
                fontfamily="sans-serif")
        ax.text(0.70, y + 0.03, auc,
                color=color, fontsize=16, ha="center", va="center",
                fontweight="bold", fontfamily="sans-serif")
        if tag:
            ax.text(0.82, y + 0.03, tag,
                    color=ORANGE, fontsize=12, ha="left", va="center",
                    fontweight="bold", fontfamily="sans-serif")

    add_accent_bar(ax, y=0.30)

    ax.text(0.5, 0.20, "XGBoost correctly ranks a churner above\n"
                        "a non-churner 87% of the time.",
            color=LIGHT_GRAY, fontsize=14, ha="center", va="center",
            fontfamily="sans-serif", linespacing=1.5)

    add_slide_number(ax, 4)
    add_branding(ax)
    return fig


# ═════════════════════════════════════════════════════════════════════
#  SLIDE 5 — Business Impact + CTA
# ═════════════════════════════════════════════════════════════════════
def slide_5():
    fig, ax = new_slide()

    ax.text(0.5, 0.90, "What This Means in Practice",
            color=WHITE, fontsize=24, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")
    add_accent_bar(ax, y=0.84)

    # recall stat
    stat_rect = mpatches.FancyBboxPatch(
        (0.12, 0.66), 0.76, 0.12, boxstyle="round,pad=0.025",
        facecolor=GREEN, edgecolor="none", alpha=0.12, transform=ax.transAxes)
    ax.add_patch(stat_rect)
    ax.text(0.5, 0.72, "At 80% recall, the model flags ~8 in 10\n"
                        "churners before they leave.",
            color=WHITE, fontsize=15, ha="center", va="center",
            fontfamily="sans-serif", linespacing=1.4)

    # revenue card
    rev_rect = mpatches.FancyBboxPatch(
        (0.12, 0.42), 0.76, 0.18, boxstyle="round,pad=0.025",
        facecolor=BG_CARD, edgecolor=ACCENT2, linewidth=1.5, transform=ax.transAxes)
    ax.add_patch(rev_rect)

    ax.text(0.5, 0.555, "Recoverable Revenue",
            color=ACCENT, fontsize=12, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    ax.text(0.5, 0.50, "£137,000 / year",
            color=ORANGE, fontsize=28, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif")

    ax.text(0.5, 0.455, "Based on £65/month avg revenue  •  1,000 customers",
            color=LIGHT_GRAY, fontsize=11, ha="center", va="center",
            fontfamily="sans-serif")

    # CTA
    cta_rect = mpatches.FancyBboxPatch(
        (0.18, 0.18), 0.64, 0.12, boxstyle="round,pad=0.025",
        facecolor=ACCENT, edgecolor="none", alpha=0.20, transform=ax.transAxes)
    ax.add_patch(cta_rect)

    ax.text(0.5, 0.24, "Full code + app on GitHub\nlink in comments ↓",
            color=ACCENT, fontsize=15, ha="center", va="center",
            fontweight="bold", fontfamily="sans-serif", linespacing=1.4)

    add_slide_number(ax, 5)
    add_branding(ax)
    return fig


# ═════════════════════════════════════════════════════════════════════
#  Build PDF
# ═════════════════════════════════════════════════════════════════════
slides = [slide_1, slide_2, slide_3, slide_4, slide_5]

with PdfPages(PDF_PATH) as pdf:
    for i, slide_fn in enumerate(slides, 1):
        fig = slide_fn()
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  Slide {i} done")

print(f"\nSaved: {PDF_PATH}")
