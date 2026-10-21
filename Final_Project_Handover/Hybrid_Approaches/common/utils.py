import sys
from pathlib import Path

import torch
from ultralytics import YOLO


# ============================================================
# Common Utilities for SugarBeet Hybrid Approaches
# ============================================================

COMMON_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = COMMON_DIR.parent.parent.parent

sys.path.insert(0, str(COMMON_DIR))

from config import (
    DATASET_70,
    DATASET_80,
    YOLO11_MODEL,
    YOLO12_MODEL,
    RTDETR_MODEL,
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
)


# ============================================================
# Device Selection
# ============================================================

def get_device():
    """
    Select the best available PyTorch device.
    """

    if torch.backends.mps.is_available():
        return "mps"

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ============================================================
# Model Loading
# ============================================================

def load_yolo11():
    """
    Load pretrained YOLOv11 nano model.
    """
    return YOLO(str(YOLO11_MODEL))


def load_yolo12():
    """
    Load pretrained YOLOv12 nano model.
    """
    return YOLO(str(YOLO12_MODEL))


def load_rtdetr():
    """
    Load pretrained RT-DETR model.
    """
    return YOLO(str(RTDETR_MODEL))


# ============================================================
# Dataset Selection
# ============================================================

def get_dataset(split_name):
    """
    Return the dataset YAML for the requested split.

    Supported:
        70_20_10
        80_10_10
    """

    if split_name == "70_20_10":
        return DATASET_70

    if split_name == "80_10_10":
        return DATASET_80

    raise ValueError(
        "Invalid split name. Use '70_20_10' or '80_10_10'."
    )


# ============================================================
# Configuration Display
# ============================================================

def print_config(
    approach_name,
    split_name,
    dataset_path,
    model_name,
):
    """
    Print a standard configuration summary.
    """

    print()
    print("=" * 70)
    print("SUGARBEET HYBRID EXPERIMENT")
    print("=" * 70)

    print("Approach :", approach_name)
    print("Split    :", split_name)
    print("Dataset  :", dataset_path)
    print("Model    :", model_name)
    print("Device   :", get_device())
    print("Image Size:", IMG_SIZE)
    print("Batch Size:", BATCH_SIZE)
    print("Epochs   :", EPOCHS)

    print("=" * 70)
    print()


# ============================================================
# Dataset Validation
# ============================================================

def validate_dataset(dataset_path):
    """
    Check whether the dataset YAML exists.
    """

    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found:\n{dataset_path}"
        )

    print("Dataset YAML verified:")
    print(dataset_path)


# ============================================================
# Model Information
# ============================================================

def print_model_info(model, model_name):
    """
    Print basic information about a loaded model.
    """

    print()
    print("-" * 60)
    print(f"{model_name} Information")
    print("-" * 60)

    print("Task   :", model.task)
    print("Classes:", model.names)

    print("-" * 60)
    print()


# ============================================================
# Main Test
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TESTING COMMON HYBRID UTILITIES")
    print("=" * 70)

    print("\nDevice:")
    print(get_device())

    print("\nDatasets:")

    validate_dataset(DATASET_70)
    validate_dataset(DATASET_80)

    print("\nLoading models...")

    yolo11 = load_yolo11()
    yolo12 = load_yolo12()
    rtdetr = load_rtdetr()

    print_model_info(yolo11, "YOLOv11")
    print_model_info(yolo12, "YOLOv12")
    print_model_info(rtdetr, "RT-DETR")

    print("\nCommon utilities test completed successfully.")
