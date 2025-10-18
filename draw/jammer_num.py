# -*- coding: utf-8 -*-
# Bar chart for Interference Type distribution (count + ratio labels)
# Aesthetic: pastel purple tone, Times New Roman, thin gray grid

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter

# ---------------- Data ----------------
class_id = np.array([0, 1, 2, 3, 4, 5, 6])
count    = np.array([576, 8244, 28608, 1180, 160, 2488, 1336])
ratio    = np.array([1.3524, 19.3557, 67.1675, 2.7705, 0.3757, 5.8415, 3.1367])  # (%)

# ---------------- Style ----------------
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 11,
})

# Soft purple palette
face_color = "#d8ccff"   # Light purple fill
edge_color = "#6b5fd9"   # Dark purple outline
grid_color = "#d0d0d0"   # Light grey grid

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.2))

bars = ax.bar(
    class_id, count,
    color=face_color,
    edgecolor=edge_color,
    linewidth=1.2
)

ax.set_xlabel("Interference Type (class_id)")
ax.set_ylabel("Sample Count")
ax.set_title("Interference Type Distribution")

# Format y-axis ticks with thousands separators and show a light grey grid
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.grid(axis="y", linestyle="--", linewidth=0.6, color=grid_color, alpha=0.8)

# Configure x-axis ticks
ax.set_xticks(class_id)
ax.set_xlim(class_id.min() - 0.6, class_id.max() + 0.6)

# Display count and ratio above each bar
for xi, c, r, rect in zip(class_id, count, ratio, bars):
    ax.annotate(
        f"{c:,} ({r:.2f}%)",
        xy=(rect.get_x() + rect.get_width()/2, rect.get_height()),
        xytext=(0, 6),
        textcoords="offset points",
        ha="center", va="bottom",
        fontsize=11, color=edge_color
    )

fig.tight_layout()

# Uncomment to save publication-friendly figures
# fig.savefig("interference_type_distribution.png", dpi=300, bbox_inches="tight")
# fig.savefig("interference_type_distribution.pdf", bbox_inches="tight")

plt.show()
