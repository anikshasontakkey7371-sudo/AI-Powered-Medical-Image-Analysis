"""
Data loading, preprocessing, verification, and augmentation module.
Handles public medical image datasets (e.g., Chest X-Ray pneumonia classification).
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Supported image file extensions
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_image_files(directory: Path) -> List[Path]:
    """Recursively find all supported image files in a directory."""
    if not directory.exists() or not directory.is_dir():
        return []
    return [
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def verify_dataset_structure(data_dir: str = "data/raw") -> Dict[str, any]:
    """
    Inspect the dataset folder and provide diagnostic information.
    Checks whether the dataset is organized in a split format (train/val/test)
    or a unified class-folder format.

    Returns a dictionary detailing status, classes, and image counts.
    """
    base_path = Path(data_dir).resolve()
    result = {
        "exists": False,
        "format": "unknown",
        "classes": [],
        "counts": {},
        "total_images": 0,
        "is_ready": False,
        "message": ""
    }

    if not base_path.exists():
        result["message"] = (
            f"Dataset directory not found: '{base_path}'.\n"
            "Please create the directory and place your chest X-ray images inside."
        )
        return result

    result["exists"] = True

    # Check for split format: train/, test/, val/ (or validation/)
    subdirs = [d.name.lower() for d in base_path.iterdir() if d.is_dir()]
    is_split = "train" in subdirs

    if is_split:
        result["format"] = "split"
        train_dir = base_path / "train"
        val_dir = (
            base_path / "val"
            if (base_path / "val").is_dir()
            else base_path / "validation"
        )
        test_dir = base_path / "test"

        splits = {"train": train_dir, "val": val_dir, "test": test_dir}
        classes_set = set()
        total = 0

        for split_name, split_path in splits.items():
            result["counts"][split_name] = {}
            if split_path.exists() and split_path.is_dir():
                for class_dir in split_path.iterdir():
                    if class_dir.is_dir() and not class_dir.name.startswith("."):
                        classes_set.add(class_dir.name)
                        count = len(get_image_files(class_dir))
                        result["counts"][split_name][class_dir.name] = count
                        total += count

        result["classes"] = sorted(list(classes_set))
        result["total_images"] = total

        train_counts = result["counts"].get("train", {})
        has_train_images = any(count > 0 for count in train_counts.values())

        if len(result["classes"]) < 2:
            result["message"] = (
                "Found less than 2 classes in the training directory. "
                "At least 2 classes (e.g., 'NORMAL' and 'PNEUMONIA') are required."
            )
        elif not has_train_images:
            result["message"] = "Found class folders, but no valid images were detected inside."
        else:
            result["is_ready"] = True
            result["message"] = f"Dataset ready (Split format). Found {total} total images across {len(result['classes'])} classes."

    else:
        # Check for unified class folders directly under data_dir
        class_dirs = [
            d for d in base_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if class_dirs:
            result["format"] = "unified"
            result["classes"] = sorted([d.name for d in class_dirs])
            total = 0
            for c_dir in class_dirs:
                count = len(get_image_files(c_dir))
                result["counts"][c_dir.name] = count
                total += count

            result["total_images"] = total
            if len(result["classes"]) < 2:
                result["message"] = (
                    "Found less than 2 class directories in data/raw. "
                    "At least 2 classes (e.g., 'NORMAL' and 'PNEUMONIA') are required."
                )
            elif total == 0:
                result["message"] = "Class folders exist, but they contain no supported image files."
            else:
                result["is_ready"] = True
                result["message"] = (
                    f"Dataset ready (Unified format). Found {total} images across "
                    f"{len(result['classes'])} classes. Automatic validation split will be used."
                )
        else:
            result["message"] = (
                f"No class folders found in '{base_path}'.\n"
                "Expected structure:\n"
                "  data/raw/train/NORMAL/ and data/raw/train/PNEUMONIA/\n"
                "  data/raw/test/NORMAL/ and data/raw/test/PNEUMONIA/\n"
                "Or simply:\n"
                "  data/raw/NORMAL/ and data/raw/PNEUMONIA/"
            )

    return result


def build_augmentation_layer():
    """
    Build a conservative data augmentation pipeline suitable for medical imaging (X-rays).
    Note: Vertical flips are omitted because human anatomical orientations
    (e.g., chest apex vs. base) are physiologically fixed.
    """
    try:
        from tensorflow import keras
        from tensorflow.keras import layers
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required to build data augmentation. "
            "Please run: pip install -r requirements.txt"
        ) from e

    return keras.Sequential(
        [
            layers.RandomFlip("horizontal", name="aug_horizontal_flip"),
            layers.RandomRotation(0.04, fill_mode="nearest", name="aug_rotation"),
            layers.RandomZoom(0.05, fill_mode="nearest", name="aug_zoom"),
            layers.RandomTranslation(0.03, 0.03, fill_mode="nearest", name="aug_translation"),
        ],
        name="data_augmentation",
    )


def load_datasets(
    data_dir: str = "data/raw",
    img_size: Tuple[int, int] = (224, 224),
    batch_size: int = 32,
    seed: int = 42,
    validation_split: float = 0.2,
):
    """
    Load training, validation, and optional test datasets from data_dir.

    Returns:
        (train_ds, val_ds, test_ds, class_names)
    """
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required for loading datasets. "
            "Please activate your virtual environment and run: pip install -r requirements.txt"
        ) from e

    status = verify_dataset_structure(data_dir)
    if not status["is_ready"]:
        raise ValueError(
            f"Cannot load dataset from '{data_dir}'. Reason: {status['message']}"
        )

    base_path = Path(data_dir).resolve()

    if status["format"] == "split":
        train_path = base_path / "train"
        val_path = (
            base_path / "val"
            if (base_path / "val").is_dir()
            else base_path / "validation"
        )
        test_path = base_path / "test"

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
        )
        class_names = train_ds.class_names

        # Check if dedicated validation folder exists and has images
                # Create a larger validation set from training images.
        print("[INFO] Creating an 80/20 training-validation split...")

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=validation_split,
            subset="training",
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_path,
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
            validation_split=validation_split,
            subset="validation",
        )

        # Check if test dataset exists
        test_has_images = (
            test_path.exists()
            and any(len(get_image_files(d)) > 0 for d in test_path.iterdir() if d.is_dir())
        )
        if test_has_images:
            test_ds = tf.keras.utils.image_dataset_from_directory(
                test_path,
               label_mode="categorical",
                image_size=img_size,
                batch_size=batch_size,
                shuffle=False,
            )
        else:
            test_ds = None

    else:
        # Unified format: split base_path into train and val
        train_ds = tf.keras.utils.image_dataset_from_directory(
            base_path,
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=validation_split,
            subset="training",
        )
        val_ds = tf.keras.utils.image_dataset_from_directory(
            base_path,
            label_mode="categorical",
            image_size=img_size,
            batch_size=batch_size,
            shuffle=True,
            seed=seed,
            validation_split=validation_split,
            subset="validation",
        )
        class_names = train_ds.class_names
        test_ds = None

    # Apply data augmentation to training data only
    aug_layer = build_augmentation_layer()
    train_ds = train_ds.map(
        lambda x, y: (aug_layer(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Performance optimization: cache and prefetch
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    if test_ds is not None:
        test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
