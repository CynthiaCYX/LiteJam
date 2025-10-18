import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe
import matplotlib as mpl

# Use Times New Roman for all English text
mpl.rcParams['font.family'] = 'Times New Roman'

# ------------------------ Axes / Categories ------------------------
categories = [
    "Interference Detection\n(F1)",
    "Interference Type\n(F1)",
    "Antenna Area\n(F1)",
    "Signal-to-Noise Ratio\n($R^2$)",
    "Bandwidth\n($R^2$)",
]
N = len(categories)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# ------------------------ Data (F1 / R^2) ------------------------
# ------------------------ Data (F1 / R^2) ------------------------
scores = {
    "MobileNetV3-Small":   [1.0000, 0.8311, 0.8259, 0.9747, 0.9845],
    "ShuffleNetV2(1.0x)":  [1.0000, 0.9137, 0.7651, 0.8484, 0.9840],
    "GhostNet 1.0":        [1.0000, 0.7123, 0.6424, 0.5250, 0.9420],
    "EfficientFormer-L1":  [1.0000, 0.8638, 0.6961, 0.9283, 0.9366],
    "MobileOne-S0":        [1.0000, 0.8744, 0.8049, 0.9660, 0.9807],
    "ResNet-18":           [1.0000, 0.9464, 0.8881, 0.9387, 0.9781],
    "LiteJam (Proposed)":  [1.0000, 0.9574, 0.9129, 0.9761, 0.9871],
}

# scores = {
#     "MobileNetV3-Small":      [1.0000, 0.6584, 0.8746, 0.8832, 0.9567],
#     "ShuffleNetV2(1.0x)":  [0.9989, 0.7715, 0.8581, 0.8650, 0.9515],
#     "GhostNet 1.0":       [1.0000, 0.5086, 0.6979, 0.5245, 0.9460],
#     "EfficientFormer-L1": [1.0000, 0.6910, 0.6439, 0.8653, 0.8922],
#     "MobileOne-S0":       [1.0000, 0.6436, 0.7441, 0.8663, 0.9478],
#     "ResNet-18":           [1.0000, 0.7637, 0.9160, 0.8595, 0.9543],
#     "LiteJam (Proposed)":     [1.0000, 0.7813, 0.9277, 0.9266, 0.9639],
# }
scores_closed = {k: v + v[:1] for k, v in scores.items()}

# ------------------------ Colors (pastel; LiteJam darkest edge) ------------------------
face_colors = {
    "MobileNetV3-Small":      "#C9CBFF",
    "ShuffleNetV2(1.0x)":  "#BFD7FF",
    "GhostNet 1.0":       "#F4C7A6",
    "EfficientFormer-L1": "#E2D0FF",
    "MobileOne-S0":       "#F6C7D4",
    "ResNet-18":           "#CDE9D6",
    "LiteJam (Proposed)":     "#B89BFF",
}
edge_colors = {
    "MobileNetV3-Small":      "#8D92F7",
    "ShuffleNetV2(1.0x)":  "#7DB5F5",
    "GhostNet 1.0":       "#D79C7E",
    "EfficientFormer-L1": "#C39EFA",
    "MobileOne-S0":       "#EE9FB4",
    "ResNet-18":           "#79C09B",
    "LiteJam (Proposed)":     "#4B33B8",
}

fill_alpha = 0.28
edge_alpha = 1.00
linewidth  = 1.0

order = [
    "MobileNetV3-Small",
    "ShuffleNetV2(1.0x)",
    "GhostNet 1.0",
    "EfficientFormer-L1",
    "MobileOne-S0",
    "ResNet-18",
    "LiteJam (Proposed)",
]

# ------------------------ Figure ------------------------
fig = plt.figure(figsize=(5.0, 4.0), dpi=170)
ax = plt.subplot(111, polar=True)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_rlabel_position(93)
ax.set_ylim(0.2, 1.0)  # radial range 0.2-1.0

# Background & grid (light purple tone)
bg_grid_color = "#DCCEEF"
outer_circle_color = "#C8B6E6"

grid_levels = [0.2, 0.4, 0.6, 0.8, 1.0]
ax.set_yticks(grid_levels)
ax.set_yticklabels([f"{g:.1f}" for g in grid_levels], fontsize=9, color="#6F6F7A")
ax.tick_params(colors="#6F6F7A")

# Category labels: keep the TOP label as-is; re-place the other four to avoid overlap (robust version)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=10, color="#333333")
stroke = [pe.withStroke(linewidth=3, foreground='white', alpha=0.95)]

# Cache tick label objects once; matplotlib may return fewer than requested in some backends
xtick_texts = list(ax.get_xticklabels())
# Apply white stroke to whatever labels exist
for t in xtick_texts:
    t.set_path_effects(stroke)

# index 0 is the TOP axis because we start at pi/2
top_index = 0

label_r = ax.get_ylim()[1] * 1.08  # place outside the outer ring
for i, (ang, lab) in enumerate(zip(angles[:-1], categories)):
    if i == top_index:
        # keep the default top label untouched
        continue

    # Hide default ticklabel if it exists, then draw our own slightly further out
    if i < len(xtick_texts):
        xtick_texts[i].set_visible(False)

    ang_deg = np.degrees(ang)
    if 80 < ang_deg < 100:
        ha, va = 'center', 'bottom'
    elif 260 < ang_deg < 280:
        ha, va = 'center', 'top'
    elif 0 < ang_deg < 180:
        ha, va = 'left', 'center'
    else:
        ha, va = 'right', 'center'

    ax.text(ang, label_r, lab,
            ha=ha, va=va,
            fontsize=10, color="#333333",
            path_effects=stroke, zorder=10)

# concentric rings
theta_full = np.linspace(0, 2*np.pi, 1024)
for g in grid_levels:
    ax.plot(theta_full, np.full_like(theta_full, g), color=bg_grid_color, linewidth=1.1, alpha=1.0, zorder=0)

# outer circle
ax.spines['polar'].set_visible(True)
ax.spines['polar'].set_color(outer_circle_color)
ax.spines['polar'].set_linewidth(1.6)

# Polygons
handles = []
for name in order:
    vals = scores_closed[name]
    ln, = ax.plot(angles, vals, color=edge_colors[name], linewidth=linewidth, alpha=edge_alpha, label=name, zorder=3)
    ax.fill(angles, vals, color=face_colors[name], alpha=fill_alpha, zorder=2, linewidth=0)
    handles.append(ln)

# Legend: frameless rectangular color swatches
legend_patches = [Rectangle((0, 0), 1, 1, facecolor=face_colors[n], edgecolor='none') for n in order]
leg = ax.legend(legend_patches, order,
                loc="lower center", bbox_to_anchor=(0.5, -0.28),
                ncol=3, frameon=False, fontsize=9,
                handlelength=1.8, columnspacing=1.2)

plt.subplots_adjust(bottom=0.30)  # reserve space for legend
plt.tight_layout()
plt.savefig("draw/radarmap_base.png", bbox_inches="tight", dpi=300)
plt.show()
