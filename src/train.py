"""
Model training pipeline for educational medical image classification.
Includes seed setting, checkpointing, early stopping, evaluation, and plot generation.
"""

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report

from src.data import load_datasets, verify_dataset_structure
from src.model import build_model
from src.plots import plot_confusion_matrix, plot_training_curves


def set_seed(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and TensorFlow for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def train_pipeline(
    data_dir: str = "data/raw",
    model_type: str = "baseline",
    epochs: int = 15,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    img_size: tuple = (224, 224),
    seed: int = 42,
    output_dir: str = "outputs",
    model_dir: str = "models",
) -> dict:
    """
    Execute full training, evaluation, and artifact generation pipeline.
    """
    set_seed(seed)

    out_path = Path(output_dir).resolve()
    model_path = Path(model_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    model_path.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  EDUCATIONAL MEDICAL IMAGE ANALYSIS: TRAINING PIPELINE")
    print("  Disclaimer: Not for clinical or diagnostic medical decisions.")
    print("=" * 65)

    # 1. Verify Dataset
    print(f"\n[1/5] Checking dataset at '{data_dir}'...")
    status = verify_dataset_structure(data_dir)
    if not status["is_ready"]:
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not ready in '{data_dir}'.\n{status['message']}\n"
            "Please review 'docs/EXECUTION_GUIDE.md' on how to download and place images."
        )

    print(f"      Status: Ready ({status['format']} structure)")
    print(f"      Detected Classes ({len(status['classes'])}): {status['classes']}")
    print(f"      Total Images: {status['total_images']}")

    # 2. Load Datasets
    print("\n[2/5] Loading and batching datasets...")
    train_ds, val_ds, test_ds, class_names = load_datasets(
        data_dir=data_dir,
        img_size=img_size,
        batch_size=batch_size,
        seed=seed,
    )

    # Save class names mapping for inference
    classes_file = model_path / "class_names.json"
    with open(classes_file, "w") as f:
        json.dump(class_names, f, indent=2)
    print(f"      Saved class label mapping to: {classes_file}")

    # 3. Build Model
    print(f"\n[3/5] Building {model_type.upper()} model...")
    input_shape = (img_size[0], img_size[1], 3)
    model = build_model(
        num_classes=len(class_names),
        input_shape=input_shape,
        model_type=model_type,
        learning_rate=learning_rate,
    )
    model.summary()

    # 4. Train Model with Callbacks
    best_model_file = model_path / "best_model.keras"
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_file),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    print(f"\n[4/5] Training for up to {epochs} epochs...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Generate Training Curves Plot
    plot_curves_path = out_path / "training_curves.png"
    plot_training_curves(history.history, output_path=str(plot_curves_path))
    print(f"      Saved training curves to: {plot_curves_path}")

    # 5. Evaluate Model
    print("\n[5/5] Evaluating model performance...")
    eval_ds = test_ds if test_ds is not None else val_ds
    eval_name = "Test" if test_ds is not None else "Validation (Test split not provided)"

    print(f"      Evaluating on {eval_name} dataset...")
    eval_results = model.evaluate(eval_ds, return_dict=True, verbose=0)
    for metric_name, val in eval_results.items():
        print(f"      - {metric_name.capitalize()}: {val:.4f}")

    # Compute Confusion Matrix and Classification Report
    y_true_list = []
    y_pred_list = []

    for images, labels in eval_ds:
        preds = model.predict(images, verbose=0)
        y_true_list.extend(np.argmax(labels.numpy(), axis=1))
        y_pred_list.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true_list)
    y_pred = np.array(y_pred_list)

    cm_path = out_path / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, output_path=str(cm_path))
    print(f"      Saved confusion matrix to: {cm_path}")

    report_str = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print("\nClassification Report:")
    print(report_str)

    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)

    # Save summary report JSON
    summary_data = {
        "timestamp": datetime.now().isoformat(),
        "disclaimer": "Educational demonstration only. Not for clinical diagnosis.",
        "model_type": model_type,
        "input_shape": list(input_shape),
        "epochs_trained": len(history.history["loss"]),
        "evaluation_dataset": eval_name,
        "evaluation_metrics": eval_results,
        "classification_report": report_dict,
        "classes": class_names,
    }
    summary_file = out_path / "training_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"      Saved training summary to: {summary_file}")

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETED SUCCESSFULLY")
    print(f"  Best model saved at: {best_model_file}")
    print("=" * 65)

    return summary_data


if __name__ == "__main__":
    train_pipeline()
