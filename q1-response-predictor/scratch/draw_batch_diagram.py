import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_linear_workflow_diagram(out_path):
    fig, ax = plt.subplots(figsize=(11, 3.2), dpi=300)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # 3-step linear pipeline nodes: (x, y, width, height, title, subtitle, bg_color, text_color)
    nodes = [
        (
            0.18, 0.50, 0.28, 0.65,
            "1. Signature Extraction",
            "Compute 6 gene signatures\nwithin each cohort matrix\n(Liu, Hugo, Riaz, TCGA)",
            "#37474F", "white"
        ),
        (
            0.50, 0.50, 0.28, 0.65,
            "2. Cohort Z-Score Scaling",
            "Standardise signatures to\nmean=0, std=1 independently\nper study cohort",
            "#009E73", "white"
        ),
        (
            0.82, 0.50, 0.28, 0.65,
            "3. Zero-Leakage LOCO ML",
            "Train ML models across trials;\ntest set remains strictly\nisolated during scaling",
            "#0072B2", "white"
        )
    ]

    # Draw Boxes
    for x, y, w, h, title, subtitle, bg, fg in nodes:
        rect = patches.FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.04",
            linewidth=1.8,
            edgecolor="#1E293B",
            facecolor=bg,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(x, y + 0.14, title, ha="center", va="center", color=fg, fontsize=11, fontweight="bold", zorder=4)
        ax.text(x, y - 0.10, subtitle, ha="center", va="center", color=fg, fontsize=9.5, fontweight="normal", zorder=4)

    # Draw Connector Arrows
    arrows = [
        ((0.32, 0.50), (0.36, 0.50)),
        ((0.64, 0.50), (0.68, 0.50))
    ]

    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops=dict(arrowstyle="-|>", color="#1E293B", lw=2.5, mutation_scale=18),
            zorder=2,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Linear workflow diagram saved to: {out_path}")

if __name__ == "__main__":
    out_file = Path(r"c:\Users\Amanda\Dropbox\OBSIDIAN\42\090 STUDY\091 UCD\091.03 ASSIGNMENTS\AI-ML-3\melanoma-assignment-3\q1-response-predictor\plots\signatures\batch_workflow_diagram.png")
    create_linear_workflow_diagram(out_file)
