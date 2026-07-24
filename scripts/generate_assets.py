"""Generate the README chart/demo assets from real project output.

Not required to run the classifier -- this is a one-off dev utility used to
produce the images committed under assets/. Requires the trained model
(run `python main.py` or `python -m src.train` first) plus matplotlib and
Pillow (`pip install matplotlib pillow`).

Usage:
    python scripts/generate_assets.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import load_dataset  # noqa: E402

ASSETS_DIR = ROOT / "assets"
METRICS_PATH = ROOT / "models" / "metrics.json"

# Reference palette (see dataviz skill / references/palette.md)
BLUE = "#2a78d6"      # categorical slot 1 -> "ham"
ORANGE = "#eb6834"    # categorical slot 2 -> "spam"
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "text.color": INK_PRIMARY,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
    }
)


def plot_class_distribution():
    df = load_dataset()
    counts = df["label"].value_counts().reindex(["ham", "spam"])

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=180)
    bars = ax.bar(counts.index, counts.values, width=0.5, color=[BLUE, ORANGE], zorder=3)

    for bar, value in zip(bars, counts.values):
        pct = value / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.015,
            f"{value:,} ({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=11,
            color=INK_PRIMARY,
        )

    ax.set_ylabel("Number of messages")
    ax.set_title("Dataset class distribution", fontsize=13, color=INK_PRIMARY, pad=14)
    ax.grid(axis="y", color=GRIDLINE, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.set_ylim(0, counts.max() * 1.15)

    fig.tight_layout()
    out = ASSETS_DIR / "class_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


def plot_confusion_matrix():
    metrics = json.loads(METRICS_PATH.read_text())
    best = metrics["best_model"]
    cm = metrics["results"][best]["confusion_matrix"]  # [[ham,ham_pred_as_spam],[spam_pred_as_ham,spam]]
    labels = ["ham", "spam"]

    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("seq_blue", SEQ_BLUE)

    # Normalize per row (recall-style) so the color encodes each class's own
    # split rather than being dominated by the much larger ham count.
    row_totals = [sum(row) for row in cm]
    cm_norm = [[cm[i][j] / row_totals[i] for j in range(2)] for i in range(2)]

    fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=180)
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            value = cm[i][j]
            share = cm_norm[i][j]
            text_color = "#ffffff" if share > 0.55 else INK_PRIMARY
            ax.text(
                j, i, f"{value}\n({share:.1%})",
                ha="center", va="center", fontsize=15, color=text_color, fontweight="bold",
            )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels([f"predicted {l}" for l in labels], fontsize=10)
    ax.set_yticklabels([f"actual {l}" for l in labels], fontsize=10)
    ax.set_title(f"Confusion matrix -- {best.replace('_', ' ').title()}", fontsize=12.5, color=INK_PRIMARY, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    fig.tight_layout()
    out = ASSETS_DIR / "confusion_matrix.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"Saved {out}")


TERMINAL_LINES = [
    ("prompt", "$ python main.py"),
    ("muted", "No trained model found - training now..."),
    ("blank", ""),
    ("text", "[naive_bayes] accuracy=0.9812  f1=0.9258"),
    ("text", "[logistic_regression] accuracy=0.9785  f1=0.9205"),
    ("blank", ""),
    ("good", "Best model: naive_bayes (F1-score=0.9258)"),
    ("muted", "Saved model to models/spam_classifier.joblib"),
    ("muted", "Saved vectorizer to models/tfidf_vectorizer.joblib"),
    ("blank", ""),
    ("rule", "=" * 58),
    ("text", "Spam Email / SMS Classifier - demo"),
    ("rule", "=" * 58),
    ("spam", '[SPAM] (87.02%)  "Congratulations! You\'ve WON a $1000 gift card..."'),
    ("ham", "[HAM]  (0.14%)   'Hey, are we still meeting for lunch tomorrow?'"),
    ("blank", ""),
    ("muted", "Type your own message below (Ctrl+C to exit)."),
    ("prompt", ">> Free entry! Text WIN to 88888 to claim your prize now"),
    ("spam", "[SPAM] (99.88%)  'Free entry! Text WIN to 88888 to claim your prize now'"),
]

TERM_BG = "#1a1a19"
TERM_CHROME = "#2c2c2a"
COLOR_MAP = {
    "prompt": "#3ecf5f",
    "muted": "#898781",
    "text": "#ffffff",
    "good": "#3ecf5f",
    "rule": "#52514e",
    "spam": "#e66767",
    "ham": "#3987e5",
    "blank": "#ffffff",
}


def _load_font(size):
    for candidate in ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/SFNSMono.ttf"):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_terminal_frame(lines_to_draw, total_lines, width=980, line_height=26, font_size=15):
    """Render a terminal mockup at a FIXED canvas size (sized for total_lines),
    drawing only the first `lines_to_draw` lines of content. Keeping every
    frame the same size is required for a well-formed animated GIF."""
    top_pad, side_pad = 56, 24
    height = top_pad + total_lines * line_height + 24
    img = Image.new("RGB", (width, height), TERM_BG)
    draw = ImageDraw.Draw(img)

    draw.rectangle([0, 0, width, 40], fill=TERM_CHROME)
    for i, color in enumerate(["#ff5f56", "#ffbd2e", "#27c93f"]):
        cx = 22 + i * 22
        draw.ellipse([cx - 6, 20 - 6, cx + 6, 20 + 6], fill=color)
    title_font = _load_font(13)
    title = "diyor@macbook -- python main.py"
    tw = draw.textlength(title, font=title_font)
    draw.text(((width - tw) / 2, 13), title, font=title_font, fill="#c3c2b7")

    font = _load_font(font_size)
    y = top_pad
    for kind, line in TERMINAL_LINES[:lines_to_draw]:
        draw.text((side_pad, y), line, font=font, fill=COLOR_MAP[kind])
        y += line_height
    return img


def make_demo_png():
    img = _draw_terminal_frame(len(TERMINAL_LINES), len(TERMINAL_LINES))
    out = ASSETS_DIR / "demo.png"
    img.save(out)
    print(f"Saved {out}")


def make_demo_gif():
    total = len(TERMINAL_LINES)
    frames = [_draw_terminal_frame(n, total) for n in range(1, total + 1)]
    durations = [500 if TERMINAL_LINES[i][0] != "blank" else 150 for i in range(len(frames))]
    durations[-1] = 3500
    out = ASSETS_DIR / "demo.gif"
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    ASSETS_DIR.mkdir(exist_ok=True)
    plot_class_distribution()
    plot_confusion_matrix()
    make_demo_png()
    make_demo_gif()
