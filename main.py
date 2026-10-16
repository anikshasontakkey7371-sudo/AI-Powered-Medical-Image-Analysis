"""
AI-Powered Medical Image Analysis - Educational Demonstration.
Main command-line entry point for dataset inspection, model training, evaluation, and prediction.

IMPORTANT DISCLAIMER:
This software is strictly intended for educational and research demonstration purposes.
It is NOT a medical diagnostic system and MUST NEVER be used for clinical decision-making.
"""

import argparse
import sys
from pathlib import Path

from src.data import verify_dataset_structure

BANNER = r"""
========================================================================
       AI-POWERED MEDICAL IMAGE ANALYSIS (EDUCATIONAL DEMO)
      A Deep Learning Baseline for Educational Study on X-rays
========================================================================
 DISCLAIMER: This tool is built purely for educational, non-clinical study.
 NOT evaluated, cleared, or approved for clinical diagnostic workflows.
========================================================================
"""


def cmd_check_data(args: argparse.Namespace) -> None:
    """Inspect dataset folder and display comprehensive readiness report."""
    print("\n--- Inspecting Dataset Directory ---")
    status = verify_dataset_structure(args.data_dir)

    print(f"Directory Path : {Path(args.data_dir).resolve()}")
    print(f"Folder Exists  : {'Yes' if status['exists'] else 'No'}")
    print(f"Format Found   : {status['format']}")
    print(f"Detected Classes: {status['classes']}")
    print(f"Total Images   : {status['total_images']}")
    print(f"Ready to Train : {'YES' if status['is_ready'] else 'NO'}")

    if status["counts"]:
        print("\nImage Count Breakdown:")
        for split_or_class, detail in status["counts"].items():
            if isinstance(detail, dict):
                print(f"  [{split_or_class}]")
                for c_name, count in detail.items():
                    print(f"    - {c_name:<15}: {count} images")
            else:
                print(f"  - {split_or_class:<15}: {detail} images")

    print(f"\nStatus Message : {status['message']}")
    if not status["is_ready"]:
        print("\nNext step: Review 'docs/EXECUTION_GUIDE.md' for dataset setup instructions.")


def cmd_train(args: argparse.Namespace) -> None:
    """Run model training and evaluation pipeline."""
    try:
        from src.train import train_pipeline
    except ImportError as e:
        print(f"\n[DEPENDENCY ERROR] Missing required libraries ({e}).")
        print("Please activate your virtual environment and install dependencies:")
        print("    .\\venv\\Scripts\\Activate.ps1")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    try:
        train_pipeline(
            data_dir=args.data_dir,
            model_type=args.model_type,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )
    except FileNotFoundError as e:
        print(f"\n[DATA ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[TRAINING ERROR] {e}")
        sys.exit(1)


def cmd_predict(args: argparse.Namespace) -> None:
    """Run inference on a single image file."""
    try:
        from src.predict import predict_single_image, print_prediction_summary
    except ImportError as e:
        print(f"\n[DEPENDENCY ERROR] Missing required libraries ({e}).")
        print("Please activate your virtual environment and install dependencies:")
        print("    .\\venv\\Scripts\\Activate.ps1")
        print("    pip install -r requirements.txt")
        sys.exit(1)

    try:
        result = predict_single_image(
            image_path=args.image,
            model_path=args.model_path,
            save_plot=not args.no_plot,
        )
        print_prediction_summary(result)
    except FileNotFoundError as e:
        print(f"\n[FILE NOT FOUND] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[PREDICTION ERROR] {e}")
        sys.exit(1)


def cmd_status(args: argparse.Namespace) -> None:
    """Display quick project workspace status."""
    data_status = verify_dataset_structure("data/raw")
    model_exists = Path("models/best_model.keras").exists()
    plots_exist = Path("outputs/training_curves.png").exists()

    print(BANNER)
    print("Project Workspace Status:")
    print(f"  1. Dataset (data/raw)    : {'Ready (' + str(data_status['total_images']) + ' images)' if data_status['is_ready'] else 'Empty / Not Configured'}")
    print(f"  2. Model (models/)       : {'Trained model found (best_model.keras)' if model_exists else 'Not trained yet'}")
    print(f"  3. Outputs (outputs/)    : {'Artifacts present' if plots_exist else 'No training plots yet'}")

    print("\nQuick Start Commands:")
    print("  - Check dataset status   : python main.py check-data")
    print("  - Train baseline model   : python main.py train --epochs 15")
    print("  - Train with MobileNetV2 : python main.py train --model-type mobilenetv2")
    print("  - Run inference on image : python main.py predict --image path/to/sample.jpeg")
    print("  - View detailed docs     : Open 'docs/EXECUTION_GUIDE.md'")
    print("=" * 72)


def build_cli_parser() -> argparse.ArgumentParser:
    """Construct argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="AI Medical Image Analysis — Educational Baseline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Command: check-data
    sub_check = subparsers.add_parser("check-data", help="Verify dataset readiness and class counts")
    sub_check.add_argument("--data-dir", type=str, default="data/raw", help="Path to raw dataset directory")
    sub_check.set_defaults(func=cmd_check_data)

    # Command: train
    sub_train = subparsers.add_parser("train", help="Train the model on the prepared dataset")
    sub_train.add_argument("--data-dir", type=str, default="data/raw", help="Path to dataset directory")
    sub_train.add_argument("--model-type", type=str, default="baseline", choices=["baseline", "mobilenetv2"], help="Model architecture")
    sub_train.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    sub_train.add_argument("--batch-size", type=int, default=32, help="Mini-batch size")
    sub_train.add_argument("--lr", type=float, default=1e-3, help="Learning rate for Adam optimizer")
    sub_train.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    sub_train.set_defaults(func=cmd_train)

    # Command: predict
    sub_pred = subparsers.add_parser("predict", help="Predict class for a single input image")
    sub_pred.add_argument("--image", type=str, required=True, help="Path to X-ray image file")
    sub_pred.add_argument("--model-path", type=str, default="models/best_model.keras", help="Path to trained model")
    sub_pred.add_argument("--no-plot", action="store_true", help="Disable generating prediction plot card")
    sub_pred.set_defaults(func=cmd_predict)

    # Command: status
    sub_status = subparsers.add_parser("status", help="Show project workspace status and quickstart guide")
    sub_status.set_defaults(func=cmd_status)

    return parser


def main() -> None:
    parser = build_cli_parser()
    if len(sys.argv) == 1:
        cmd_status(None)
    else:
        args = parser.parse_args()
        if hasattr(args, "func"):
            args.func(args)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
