"""
Inference module for single-image and batch predictions.
Evaluates medical chest X-ray images and reports confidence scores.

IMPORTANT: Educational demonstration only. Not for clinical diagnosis.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

from src.plots import plot_prediction_card

DISCLAIMER_NOTICE = (
    "----------------------------------------------------------------------\n"
    " [!] EDUCATIONAL DEMONSTRATION ONLY\n"
    " This AI tool is built for learning and demonstration purposes.\n"
    " It must NOT be used to diagnose, triage, or treat any medical condition.\n"
    " Always consult a certified healthcare professional for medical advice.\n"
    "----------------------------------------------------------------------"
)


def load_model_and_classes(
    model_path: str = "models/best_model.keras",
    classes_path: str = "models/class_names.json",
) -> Tuple[keras.Model, List[str]]:
    """
    Safely load a trained Keras model and its associated class labels.
    """
    model_file = Path(model_path).resolve()
    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained model not found at '{model_file}'.\n"
            "Please train the model first by running:\n"
            "    python main.py train\n"
            "or review 'docs/EXECUTION_GUIDE.md' for complete instructions."
        )

    model = keras.models.load_model(model_file)

    classes_file = Path(classes_path).resolve()
    if classes_file.exists():
        with open(classes_file, "r") as f:
            class_names = json.load(f)
    else:
        # Fallback to standard 2-class baseline if mapping file is absent
        class_names = ["NORMAL", "PNEUMONIA"]

    return model, class_names


def preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, Path]:
    """
    Validate and preprocess an input image file for model inference.
    """
    img_file = Path(image_path).resolve()

    if not img_file.exists():
        raise FileNotFoundError(f"Input image not found: '{img_file}'")

    if not img_file.is_file():
        raise ValueError(f"Target path is not a file: '{img_file}'")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if img_file.suffix.lower() not in valid_exts:
        raise ValueError(
            f"Unsupported file format '{img_file.suffix}'. "
            f"Supported extensions: {sorted(list(valid_exts))}"
        )

    # Load and resize image to expected network input size
    img = keras.utils.load_img(img_file, target_size=target_size)
    img_array = keras.utils.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)  # Shape: (1, height, width, 3)

    return img_batch, img_file


def predict_single_image(
    image_path: str,
    model_path: str = "models/best_model.keras",
    classes_path: str = "models/class_names.json",
    save_plot: bool = True,
    output_plot_path: str = "outputs/prediction_result.png",
) -> Dict[str, any]:
    """
    Run inference on a single medical image and return prediction details.
    """
    model, class_names = load_model_and_classes(model_path, classes_path)
    img_batch, img_file = preprocess_image(image_path)

    # Execute inference
    predictions = model.predict(img_batch, verbose=0)[0]

    predicted_idx = int(np.argmax(predictions))
    predicted_class = class_names[predicted_idx]
    confidence = float(predictions[predicted_idx])

    probabilities = {
        cls: float(prob)
        for cls, prob in zip(class_names, predictions)
    }

    plot_file = None
    if save_plot:
        plot_file = plot_prediction_card(
            image_path=str(img_file),
            predicted_class=predicted_class,
            confidence=confidence,
            probabilities=probabilities,
            output_path=output_plot_path,
        )

    return {
        "image_file": str(img_file),
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "plot_saved_to": plot_file,
    }


def print_prediction_summary(result: Dict[str, any]) -> None:
    """Print a clean CLI summary of the prediction result."""
    print(DISCLAIMER_NOTICE)
    print(f"\nTarget Image : {result['image_file']}")
    print(f"Prediction   : {result['predicted_class']}")
    print(f"Confidence   : {result['confidence'] * 100:.2f}%")
    print("\nClass Probabilities Breakdown:")
    for cls, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  - {cls:<12}: {prob * 100:6.2f}%  [{bar:<30}]")

    if result.get("plot_saved_to"):
        print(f"\nVisual diagnostic card saved at: {result['plot_saved_to']}")
    print("-" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Medical Image Prediction Tool")
    parser.add_argument("--image", type=str, required=True, help="Path to input X-ray image")
    parser.add_argument("--model", type=str, default="models/best_model.keras", help="Path to trained model")
    args = parser.parse_args()

    res = predict_single_image(args.image, model_path=args.model)
    print_prediction_summary(res)
