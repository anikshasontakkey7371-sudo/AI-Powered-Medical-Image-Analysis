"""
Model architecture definitions and builder factory.
Provides both a beginner-friendly Custom Baseline CNN and a Transfer Learning (MobileNetV2) option.
"""

from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_custom_cnn(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.4,
) -> keras.Model:
    """
    Build a clean, transparent Convolutional Neural Network (CNN) baseline.
    Ideal for beginners studying how feature extraction layers work.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # Rescale pixel values from [0, 255] to [0.0, 1.0]
    x = layers.Rescaling(1.0 / 255.0, name="rescaling")(inputs)

    # Block 1 - Low level edge and gradient features
    x = layers.Conv2D(32, (3, 3), padding="same", name="conv1")(x)
    x = layers.BatchNormalization(name="bn1")(x)
    x = layers.ReLU(name="relu1")(x)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Block 2 - Textures and corner patterns
    x = layers.Conv2D(64, (3, 3), padding="same", name="conv2")(x)
    x = layers.BatchNormalization(name="bn2")(x)
    x = layers.ReLU(name="relu2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Block 3 - Intermediate anatomical silhouettes
    x = layers.Conv2D(128, (3, 3), padding="same", name="conv3")(x)
    x = layers.BatchNormalization(name="bn3")(x)
    x = layers.ReLU(name="relu3")(x)
    x = layers.MaxPooling2D((2, 2), name="pool3")(x)

    # Block 4 - High-level semantic tissue density representations
    x = layers.Conv2D(256, (3, 3), padding="same", name="conv4")(x)
    x = layers.BatchNormalization(name="bn4")(x)
    x = layers.ReLU(name="relu4")(x)
    x = layers.MaxPooling2D((2, 2), name="pool4")(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.ReLU(name="relu_dense")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="classification_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MedicalImage_BaselineCNN")
    return model


def build_mobilenetv2_transfer(
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    num_classes: int = 2,
    dropout_rate: float = 0.3,
    freeze_base: bool = True,
) -> keras.Model:
    """
    Build a Transfer Learning model utilizing MobileNetV2 pretrained on ImageNet.
    Demonstrates leveraging pretrained visual representations for medical classification.
    """
    inputs = keras.Input(shape=input_shape, name="input_image")

    # MobileNetV2 expects inputs in [-1, 1]
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = not freeze_base

    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = layers.Dropout(dropout_rate, name="transfer_dropout")(x)

    outputs = layers.Dense(
        num_classes, activation="softmax", name="classification_output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MedicalImage_MobileNetV2")
    return model


def build_model(
    num_classes: int = 2,
    input_shape: Tuple[int, int, int] = (224, 224, 3),
    model_type: str = "baseline",
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Factory function to construct and compile a model.
    """
    if model_type.lower() == "baseline":
        model = build_custom_cnn(input_shape=input_shape, num_classes=num_classes)
    elif model_type.lower() in {"mobilenet", "mobilenetv2", "transfer"}:
        model = build_mobilenetv2_transfer(
            input_shape=input_shape, num_classes=num_classes
        )
    else:
        raise ValueError(
            f"Unknown model_type '{model_type}'. Choose 'baseline' or 'mobilenetv2'."
        )

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy()
    metrics = [
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc"),
    ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model
