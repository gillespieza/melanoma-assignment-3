import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path for importing src modules
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

# Paths
PLOT_DIR = BASE_DIR / "plots" / "clinical"
PLOT_DIR.mkdir(exist_ok=True, parents=True)

from src.styles import RESPONSE_PALETTE, set_presentation_style

set_presentation_style()

def draw_waffle_cohort(ax, n_resp, n_nr, title, n_cols=10):
    total = n_resp + n_nr
    # Colors: Okabe-Ito Bluish Green for Responders, Vermillion for Non-responders
    c_resp = RESPONSE_PALETTE["CR/PR"]
    c_nr = RESPONSE_PALETTE["PD"]
    
    # List of colors for each block (Responders first, then Non-responders)
    block_colors = [c_resp] * n_resp + [c_nr] * n_nr
    
    n_rows = int(np.ceil(total / n_cols))
    
    # Configure axes limits
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)
    
    # Draw squares
    for idx in range(total):
        # We fill from bottom-left to top-right
        row = idx // n_cols
        col = idx % n_cols
        color = block_colors[idx]
        
        rect = plt.Rectangle(
            (col - 0.4, row - 0.4), 0.8, 0.8,
            facecolor=color, edgecolor='white', linewidth=1.5
        )
        ax.add_patch(rect)
        
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add subtitle with counts
    pct_resp = (n_resp / total) * 100
    subtext = f"Responders: {n_resp} ({pct_resp:.1f}%)\nNon-responders: {n_nr}"
    ax.text(n_cols / 2 - 0.5, n_rows - 0.1, f"{title} (N={total})", 
            ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(n_cols / 2 - 0.5, -0.6, subtext, 
            ha='center', va='top', fontsize=11, fontstyle='italic', linespacing=1.3)

def main():
    print("==================================================")
    print("Generating Waffle Chart for Presentation")
    print("==================================================\n")

    # Cohort data
    cohorts = {
        "Liu 2019": {"resp": 48, "nr": 56, "cols": 10},
        "Hugo 2016": {"resp": 13, "nr": 13, "cols": 10},
        "Riaz 2017": {"resp": 5, "nr": 15, "cols": 10}
    }

    # Set up figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 7))
    
    # Draw each cohort
    draw_waffle_cohort(axes[0], cohorts["Liu 2019"]["resp"], cohorts["Liu 2019"]["nr"], "Liu 2019", n_cols=10)
    draw_waffle_cohort(axes[1], cohorts["Hugo 2016"]["resp"], cohorts["Hugo 2016"]["nr"], "Hugo 2016", n_cols=5)
    draw_waffle_cohort(axes[2], cohorts["Riaz 2017"]["resp"], cohorts["Riaz 2017"]["nr"], "Riaz 2017", n_cols=5)

    # Main Title
    fig.suptitle("Patient Response Distribution (1 block = 1 patient)", 
                 fontsize=18, fontweight='bold', y=0.95)

    # Custom Legend
    import matplotlib.patches as mpatches
    legend_patches = [
        mpatches.Patch(color=RESPONSE_PALETTE["CR/PR"], label='Responder (CR/PR)'),
        mpatches.Patch(color=RESPONSE_PALETTE["PD"], label='Non-responder (PD)')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=2, fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.05))

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2, top=0.8)

    out_path = PLOT_DIR / "response_waffle_chart.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"Saved waffle chart to {out_path}")

if __name__ == "__main__":
    main()
