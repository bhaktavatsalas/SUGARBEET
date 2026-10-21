#!/usr/bin/env python3

import sys
import json
from pathlib import Path

from ultralytics import RTDETR

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
    evaluate_prf,
    calculate_map,
    save_predictions,
)

# ============================================================
# MGKD CHECKPOINT
# ============================================================

MODEL = (
    BASE
    / "Experiments/Hybrid_Approaches/MGKD/results_control/"
      "MGKD_YOLOv11_RTDETR_70_20_10_CONTROL/best.pt"
)

DATASET = BASE / "dataset_70_20_10_detection"

IMAGES = DATASET / "images" / "test"
LABELS = DATASET / "labels" / "test"

OUT = (
    BASE
    / "Experiments/Unified_Standardized_Evaluation/"
      "MGKD_YOLOv11_RTDETR_70_20_10_CONTROL"
)

# ============================================================
# FILTER PREDICTIONS FOR OPERATING CONFIDENCE
# ============================================================

def filter_predictions(predictions, confidence):

    filtered = []

    for image_preds in predictions:

        current = []

        for pred in image_preds:

            # Format:
            # [x1, y1, x2, y2, confidence, class]
            if float(pred[4]) >= confidence:
                current.append(pred)

        filtered.append(current)

    return filtered


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CONTROL — CORRECTED UNIFIED STANDARDIZED EVALUATION")
    print("=" * 80)

    print("Checkpoint :", MODEL)
    print("Dataset    :", DATASET)
    print("Image size :", IMAGE_SIZE)
    print("Raw conf   :", PRED_CONF)
    print("P/R/F1 conf:", EVAL_CONF)
    print("mAP conf   :", PRED_CONF)
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

    # --------------------------------------------------------
    # Test images
    # --------------------------------------------------------

    image_files = get_images(DATASET)

    print()
    print(f"Test images found: {len(image_files)}")

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    ground_truths = []

    for image_path in image_files:

        label_path = LABELS / f"{image_path.stem}.txt"

        ground_truths.append(
            load_ground_truth(label_path)
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print()
    print("Loading CONTROL RT-DETR checkpoint...")

    model = RTDETR(str(MODEL))

    print("Control model loaded successfully.")

    # --------------------------------------------------------
    # Raw predictions at 0.001
    # --------------------------------------------------------

    predictions = []

    print()
    print("Generating predictions...")

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        preds = get_model_predictions(
            model,
            image_path
        )

        predictions.append(preds)

        print(
            f"Processing {index}/{len(image_files)}",
            end="\r"
        )

    print()
    print("Predictions generated.")

    # --------------------------------------------------------
    # Save raw predictions
    # --------------------------------------------------------

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    raw_path = (
        OUT /
        "RT_DETR_CONTROL_raw_predictions.json"
    )

    save_predictions(
        predictions,
        image_files,
        raw_path
    )

    # --------------------------------------------------------
    # P/R/F1 at confidence = 0.25
    # --------------------------------------------------------

    operating_predictions = filter_predictions(
        predictions,
        EVAL_CONF
    )

    p, r, f1, tp, fp, fn = evaluate_prf(
        operating_predictions,
        ground_truths
    )

    # --------------------------------------------------------
    # mAP at raw confidence = 0.001
    # --------------------------------------------------------

    map50, map5095 = calculate_map(
        predictions,
        ground_truths
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("FINAL CONTROL STANDARDIZED RESULTS")
    print("=" * 80)

    print(f"Precision : {p:.4f}")
    print(f"Recall    : {r:.4f}")
    print(f"F1        : {f1:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")
    print(f"TP / FP / FN : {tp} / {fp} / {fn}")

    # --------------------------------------------------------
    # Save final result
    # --------------------------------------------------------

    result = {
        "model": "RT-DETR CONTROL",
        "split": "70_20_10",
        "test_images": len(image_files),
        "checkpoint": str(MODEL),
        "image_size": IMAGE_SIZE,
        "raw_prediction_confidence": PRED_CONF,
        "operating_confidence": EVAL_CONF,
        "evaluation_iou": EVAL_IOU,
        "precision": p,
        "recall": r,
        "f1": f1,
        "mAP50": map50,
        "mAP50_95": map5095,
        "TP": tp,
        "FP": fp,
        "FN": fn,
    }

    json_path = (
        OUT /
        "CONTROL_standardized_results_corrected.json"
    )

    txt_path = (
        OUT /
        "CONTROL_standardized_results_corrected.txt"
    )

    json_path.write_text(
        json.dumps(
            result,
            indent=2
        )
    )

    txt_path.write_text(
        "\n".join([
            "Model: RT-DETR CONTROL",
            "Split: 70_20_10",
            f"Test images: {len(image_files)}",
            f"Checkpoint: {MODEL}",
            f"Image size: {IMAGE_SIZE}",
            f"Raw prediction confidence: {PRED_CONF}",
            f"Operating confidence: {EVAL_CONF}",
            f"Evaluation IoU: {EVAL_IOU}",
            "",
            "P/R/F1 evaluated at confidence 0.25",
            "mAP50/mAP50-95 evaluated from predictions at confidence 0.001",
            "",
            f"Precision: {p:.6f}",
            f"Recall: {r:.6f}",
            f"F1: {f1:.6f}",
            f"mAP50: {map50:.6f}",
            f"mAP50-95: {map5095:.6f}",
            f"TP: {tp}",
            f"FP: {fp}",
            f"FN: {fn}",
        ]) + "\n"
    )

    print()
    print("Saved corrected results:")
    print(txt_path)

    print()
    print("=" * 80)
    print("CORRECTED CONTROL EVALUATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
