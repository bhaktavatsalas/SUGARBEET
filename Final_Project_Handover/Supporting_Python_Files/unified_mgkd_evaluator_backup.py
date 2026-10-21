#!/usr/bin/env python3

import sys
import json
from pathlib import Path

import numpy as np
from ultralytics import RTDETR

# Use the EXACT helper functions from the existing unified evaluator.
BASE = Path("/Users/kalamanjunatha/Downloads/sugarbeet")
EXPERIMENTS_DIR = BASE / "Experiments"

sys.path.insert(0, str(EXPERIMENTS_DIR))

from unified_wbf_evaluator import (
    IMAGE_SIZE,
    PRED_CONF,
    EVAL_CONF,
    EVAL_IOU,
    get_images,
    load_ground_truth,
    get_model_predictions,
    evaluate_model,
    save_predictions,
)

# ============================================================
# MGKD CHECKPOINT
# ============================================================

MODEL = (
    BASE
    / "Experiments/Hybrid_Approaches/MGKD/results_50epoch/"
      "MGKD_YOLOv11_RTDETR_70_20_10/best.pt"
)

DATASET = BASE / "dataset_70_20_10_detection"

IMAGES = DATASET / "images" / "test"
LABELS = DATASET / "labels" / "test"

OUT = (
    BASE
    / "Experiments/Unified_Standardized_Evaluation/"
      "MGKD_YOLOv11_RTDETR_70_20_10"
)

# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("MGKD — UNIFIED STANDARDIZED EVALUATION")
    print("=" * 80)

    print("Checkpoint :", MODEL)
    print("Dataset    :", DATASET)
    print("Test images:", IMAGES)
    print("Image size :", IMAGE_SIZE)
    print("Raw conf   :", PRED_CONF)
    print("Eval conf  :", EVAL_CONF)
    print("Eval IoU   :", EVAL_IOU)

    if not MODEL.exists():
        raise FileNotFoundError(
            f"MGKD checkpoint not found:\n{MODEL}"
        )

    if not IMAGES.exists():
        raise FileNotFoundError(
            f"Test image directory not found:\n{IMAGES}"
        )

    if not LABELS.exists():
        raise FileNotFoundError(
            f"Test label directory not found:\n{LABELS}"
        )

    image_files = get_images(DATASET)

    print()
    print(f"Test images found: {len(image_files)}")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truths = []

    for image_path in image_files:

        label_path = (
            LABELS /
            f"{image_path.stem}.txt"
        )

        ground_truths.append(
            load_ground_truth(label_path)
        )

    # --------------------------------------------------------
    # Load MGKD model
    # --------------------------------------------------------

    print()
    print("Loading MGKD RT-DETR checkpoint...")

    model = RTDETR(
        str(MODEL)
    )

    print("Model loaded successfully.")

    # --------------------------------------------------------
    # Generate predictions using the SAME evaluator helper
    # --------------------------------------------------------

    predictions = []

    print()
    print("Generating raw predictions...")

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        preds = get_model_predictions(
            model,
            image_path
        )

        predictions.append(
            preds
        )

        print(
            f"Processing {index}/{len(image_files)}",
            end="\r"
        )

    print()
    print("Raw predictions generated.")

    # --------------------------------------------------------
    # Save raw predictions
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_path = (
        OUT /
        "MGKD_YOLOv11_RTDETR_raw_predictions.json"
    )

    save_predictions(
        predictions,
        image_files,
        raw_path
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    result = evaluate_model(
        predictions,
        ground_truths
    )

    print()
    print("=" * 80)
    print("FINAL MGKD STANDARDIZED RESULTS")
    print("=" * 80)

    print(
        f"Precision : {result['precision']:.4f}"
    )

    print(
        f"Recall    : {result['recall']:.4f}"
    )

    print(
        f"F1        : {result['f1']:.4f}"
    )

    print(
        f"mAP50     : {result['map50']:.4f}"
    )

    print(
        f"mAP50-95  : {result['map50_95']:.4f}"
    )

    print(
        f"TP / FP / FN : "
        f"{result['TP']} / "
        f"{result['FP']} / "
        f"{result['FN']}"
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    final_result = {
        "model": "MGKD YOLOv11 -> RT-DETR",
        "split": "70_20_10",
        "test_images": len(image_files),
        "checkpoint": str(MODEL),
        "image_size": IMAGE_SIZE,
        "prediction_confidence": PRED_CONF,
        "evaluation_confidence": EVAL_CONF,
        "evaluation_iou": EVAL_IOU,
        **result
    }

    json_path = (
        OUT /
        "MGKD_standardized_results.json"
    )

    txt_path = (
        OUT /
        "MGKD_standardized_results.txt"
    )

    json_path.write_text(
        json.dumps(
            final_result,
            indent=2
        )
    )

    txt_path.write_text(
        "\n".join([
            "Model: MGKD YOLOv11 -> RT-DETR",
            "Split: 70_20_10",
            f"Test images: {len(image_files)}",
            f"Checkpoint: {MODEL}",
            f"Image size: {IMAGE_SIZE}",
            f"Prediction confidence: {PRED_CONF}",
            f"Evaluation confidence: {EVAL_CONF}",
            f"Evaluation IoU: {EVAL_IOU}",
            f"Precision: {result['precision']:.6f}",
            f"Recall: {result['recall']:.6f}",
            f"F1: {result['f1']:.6f}",
            f"mAP50: {result['map50']:.6f}",
            f"mAP50-95: {result['map50_95']:.6f}",
            f"TP: {result['TP']}",
            f"FP: {result['FP']}",
            f"FN: {result['FN']}",
        ]) + "\n"
    )

    print()
    print("Saved raw predictions:")
    print(raw_path)

    print()
    print("Saved results:")
    print(txt_path)

    print()
    print("=" * 80)
    print("MGKD EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
