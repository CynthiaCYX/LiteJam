# Plot a styled confusion matrix heatmap.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

def parse_confusion_matrix_str(matrix_str):
    """
    Convert a string representation of a confusion matrix into a numpy array,
    removing brackets and extra whitespace automatically.
    """
    # Remove outer brackets and split by line
    matrix_str = matrix_str.strip().replace('[', '').replace(']', '')
    lines = matrix_str.strip().split('\n')
    matrix = []
    for line in lines:
        # Extract integer values per row
        numbers = [int(n) for n in line.strip().split()]
        matrix.append(numbers)
    return np.array(matrix)


def plot_confusion_matrix(cm, class_names=None, title="Confusion Matrix", save_path=None):
    num_classes = cm.shape[0]

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'figure.dpi': 300
    })

    plt.figure(figsize=(0.8 * num_classes + 2, 0.7 * num_classes + 2))
    fmt = ".2f" if np.max(cm) <= 1.0 else "d"

    ax = sns.heatmap(cm,
                     annot=True,
                     fmt=fmt,
                     cmap="Blues",
                     linewidths=0.5,
                     cbar=True,
                     xticklabels=class_names,
                     yticklabels=class_names,
                     annot_kws={"size": 12})

    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(2)
        spine.set_color("black")

    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_title(title, pad=15)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=300, bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    # ==== Paste the confusion matrix string you want to visualise ====
    matrix_str = """
[[ 80   2   1   1   0   0]
 [  4  92  58   6   1   7]
 [  2   3 218   3   1   1]
 [  1   2  15  10   0   4]
 [  0   1   2   1  20   0]
 [ 22   5  78   4   1 154]]
 """

    # ==== Class labels (edit as needed) ====
    # class_labels = ["-10", "-8", "-6", "-4", "-2", "0", "2", "4", "6", "8", "10"]
    class_labels = ['None','Chirp','Freq-Hopper','Modulated','Multitone','Pulsed','Noise']
    # class_labels = ["0","1","2","3"]
    # class_labels = ["0", "1"]

    # ==== Render the plot ====
    conf_matrix = parse_confusion_matrix_str(matrix_str)
    plot_confusion_matrix(conf_matrix, class_names=class_labels, title='', save_path="draw/EfficientFormer-L1")
