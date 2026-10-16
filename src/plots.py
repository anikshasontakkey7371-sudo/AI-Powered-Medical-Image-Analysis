"""
Visualization module for educational medical image analysis.
Plots training history, confusion matrices, and prediction cards.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Global styling configuration for clean, educational presentation
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
DISCLAIMER_TEXT = "Educational Demonstration Only — Not for Clinical Diagnostic Use"


def plot_training_curves(
    history: dict,
    output_path: str = "outputs/training_curves.png",
) -> str:
    """
    Plot training and validation Loss, Accuracy, and AUC metrics across epochs.
    Saves figure to output_path and returns the saved file path.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history.get("loss", [])) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Loss Curve
    axes[0].plot(epochs, history.get("loss", []), "o-", label="Train Loss", color="#1f77b4", linewidth=2)
    if "val_loss" in history:
        axes[0].plot(epochs, history.get("val_loss", []), "s--", label="Val Loss", color="#d62728", linewidth=2)
    axes[0].set_title("Cross-Entropy Loss vs. Epochs", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Accuracy Curve
    axes[1].plot(epochs, history.get("accuracy", []), "o-", label="Train Accuracy", color="#2ca02c", linewidth=2)
    if "val_accuracy" in history:
        axes[1].plot(epochs, history.get("val_accuracy", []), "s--", label="Val Accuracy", color="#ff7f0e", linewidth=2)
    axes[1].set_title("Accuracy vs. Epochs", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim([0.0, 1.05])
    axes[1].legend(loc="lower right")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # 3. AUC Curve (or Recall)
    metric_key = "auc" if "auc" in history else ("recall" if "recall" in history else None)
    val_metric_key = f"val_{metric_key}" if metric_key else None

    if metric_key and metric_key in history:
        axes[2].plot(epochs, history[metric_key], "o-", label=f"Train {metric_key.upper()}", color="#9467bd", linewidth=2)
        if val_metric_key and val_metric_key in history:
            axes[2].plot(epochs, history[val_metric_key], "s--", label=f"Val {metric_key.upper()}", color="#8c564b", linewidth=2)
        axes[2].set_title(f"{metric_key.upper()} vs. Epochs", fontsize=12, fontweight="bold")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel(metric_key.upper())
        axes[2].set_ylim([0.0, 1.05])
        axes[2].legend(loc="lower right")
        axes[2].grid(True, linestyle="--", alpha=0.6)
    else:
        axes[2].text(0.5, 0.5, "Additional metric not tracked", ha="center", va="center")
        axes[2].axis("off")

    fig.suptitle(f"Model Training Progression\n[{DISCLAIMER_TEXT}]", fontsize=13, color="#555555")
    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str = "outputs/confusion_matrix.png",
) -> str:
    """
    Plot and save an annotated Confusion Matrix heatmap showing counts and percentages.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = np.divide(cm, cm_sum, out=np.zeros_like(cm, dtype=float), where=cm_sum != 0) * 100

    labels = np.asarray([
        f"{count}\n({perc:.1f}%)"
        for count, perc in zip(cm.flatten(), cm_perc.flatten())
    ]).reshape(cm.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=labels,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
        annot_kws={"size": 11, "weight": "bold"},
    )

    ax.set_title(f"Confusion Matrix\n[{DISCLAIMER_TEXT}]", fontsize=12, pad=12, color="#444444")
    ax.set_ylabel("True Anatomical / Clinical Label", fontsize=11, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)


def plot_prediction_card(
    image_path: str,
    predicted_class: str,
    confidence: float,
    probabilities: Dict[str, float],
    output_path: str = "outputs/prediction_result.png",
) -> str:
    """
    Generate an educational diagnostic preview card displaying the X-ray
    alongside a probability bar chart and mandatory educational disclaimer.
    """
    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    img = plt.imread(image_path)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.2, 1])

    # Left: Image display
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(img, cmap="gray" if img.ndim == 2 else None)
    ax_img.set_title(Path(image_path).name, fontsize=11)
    ax_img.axis("off")

    # Right: Probability distribution
    ax_bar = fig.add_subplot(gs[0, 1])
    classes = list(probabilities.keys())
    probs = [probabilities[c] * 100 for c in classes]
    colors = ["#d62728" if c == "PNEUMONIA" else "#1f77b4" for c in classes]

    y_pos = np.arange(len(classes))
    ax_bar.barh(y_pos, probs, color=colors, height=0.5, alpha=0.85)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(classes, fontsize=11, fontweight="bold")
    ax_bar.set_xlim([0, 100])
    ax_bar.set_xlabel("Confidence (%)", fontsize=10)
    ax_bar.set_title(f"Prediction: {predicted_class} ({confidence*100:.1f}%)", fontsize=12, fontweight="bold")

    for i, p in enumerate(probs):
        ax_bar.text(p + 1.5, i, f"{p:.1f}%", va="center", fontsize=10, fontweight="bold")

    fig.suptitle(
        f"Model Prediction Analysis\n[{DISCLAIMER_TEXT}]",
        fontsize=11,
        color="#c0392b",
        fontweight="bold",
    )

    plt.tight_layout()
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_file)
