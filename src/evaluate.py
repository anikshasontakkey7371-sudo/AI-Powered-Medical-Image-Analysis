"""Evaluate the saved model on the test dataset; educational use only."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report
from src.data import load_datasets
from src.predict import load_model_and_classes
from src.plots import plot_confusion_matrix


def main():
    output = Path("outputs")
    output.mkdir(exist_ok=True)
    _, _, test_ds, dataset_classes = load_datasets(batch_size=32, seed=42)
    if test_ds is None:
        raise RuntimeError("No test dataset found.")
    model, classes = load_model_and_classes()
    if dataset_classes != classes:
        raise ValueError(f"Class order mismatch: {dataset_classes} vs {classes}")
    print("Evaluating saved model; no training will run.")
    metrics = model.evaluate(test_ds, return_dict=True, verbose=1)
    truth, predicted = [], []
    for images, labels in test_ds:
        scores = model.predict(images, verbose=0)
        truth.extend(np.argmax(labels.numpy(), axis=1))
        predicted.extend(np.argmax(scores, axis=1))
    report = classification_report(truth, predicted, labels=list(range(len(classes))), target_names=classes, output_dict=True, zero_division=0)
    print("\nTest metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(truth, predicted, labels=list(range(len(classes))), target_names=classes, zero_division=0))
    plot_confusion_matrix(np.array(truth), np.array(predicted), classes, output_path=str(output / "confusion_matrix.png"))
    summary = {"disclaimer": "Educational use only; not for clinical diagnosis.", "evaluation_dataset": "test", "test_images": len(truth), "metrics": {k: float(v) for k, v in metrics.items()}, "classification_report": report, "classes": classes}
    (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nSaved outputs/confusion_matrix.png and outputs/evaluation_summary.json")

if __name__ == "__main__":
    main()
