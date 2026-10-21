import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# GLOBAL SETTINGS
# ============================================================

IMAGE_SIZE = 640

# Prediction generation threshold
PRED_CONF = 0.001

# Final evaluation confidence threshold
EVAL_CONF = 0.25

# IoU used for TP/FP/FN and AP
EVAL_IOU = 0.50

# IoU used by WBF
WBF_IOU = 0.55

# WBF candidate filtering
WBF_SKIP = 0.05

OUTPUT_ROOT = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
    "Unified_Standardized_Evaluation"
)

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATASETS
# ============================================================

DATASETS = {
    "70_20_10": {
        "root": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/"
            "dataset_70_20_10_detection"
        ),
        "yolo11": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
            "70_20_10/YOLOv11_70_20_10_Detection/"
            "weights/best.pt"
        ),
        "yolo12": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
            "70_20_10/YOLOv12_70_20_10_Detection/train/"
            "weights/best.pt"
        ),
        "rtdetr": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
            "70_20_10/RT_DETR_70_20_10/"
            "weights/best.pt"
        ),
    },

    "80_10_10": {
        "root": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/"
            "dataset_80_10_10_detection"
        ),
        "yolo11": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
            "80_10_10/YOLOv11/detection/"
            "weights/best.pt"
        ),
        "yolo12": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
            "80_10_10/YOLOv12/detection-2/"
            "weights/best.pt"
        ),
        "rtdetr": Path(
            "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/"
            "80_10_10/RT_DETR4/"
            "weights/best.pt"
        ),
    }
}


# ============================================================
# IMAGE DISCOVERY
# ============================================================

def get_images(root):

    image_dir = root / "images" / "test"

    images = sorted(
        list(image_dir.glob("*.jpg")) +
        list(image_dir.glob("*.jpeg")) +
        list(image_dir.glob("*.png"))
    )

    return images


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(label_path):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, parts)

            x1 = xc - bw / 2
            y1 = yc - bh / 2
            x2 = xc + bw / 2
            y2 = yc + bh / 2

            boxes.append([
                x1,
                y1,
                x2,
                y2
            ])

    return boxes


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_model_predictions(model, image_path):

    result = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=PRED_CONF,
        iou=0.7,
        verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)

    h, w = result.orig_shape

    predictions = []

    for box, score, label in zip(
        boxes,
        scores,
        labels
    ):

        x1, y1, x2, y2 = box

        # Normalize to 0-1
        x1 /= w
        x2 /= w
        y1 /= h
        y2 /= h

        predictions.append([
            float(x1),
            float(y1),
            float(x2),
            float(y2),
            float(score),
            int(label)
        ])

    return predictions


# ============================================================
# CONFIDENCE FILTER
# ============================================================

def filter_predictions(predictions):

    return [
        p for p in predictions
        if p[4] >= EVAL_CONF
    ]


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def evaluate_prf(
    predictions,
    ground_truths
):

    TP = 0
    FP = 0
    FN = 0

    for preds, gts in zip(
        predictions,
        ground_truths
    ):

        preds = filter_predictions(preds)

        matched = set()

        # Highest confidence first
        preds = sorted(
            preds,
            key=lambda x: x[4],
            reverse=True
        )

        for pred in preds:

            best_iou = 0.0
            best_gt = -1

            for i, gt in enumerate(gts):

                if i in matched:
                    continue

                iou = calculate_iou(
                    pred[:4],
                    gt
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt = i

            if (
                best_gt >= 0
                and best_iou >= EVAL_IOU
            ):

                TP += 1
                matched.add(best_gt)

            else:

                FP += 1

        FN += len(gts) - len(matched)

    precision = (
        TP / (TP + FP)
        if TP + FP > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0
        else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0
    )

    return (
        precision,
        recall,
        f1,
        TP,
        FP,
        FN
    )


# ============================================================
# AP
# ============================================================

def calculate_ap(
    predictions,
    ground_truths,
    iou_threshold
):

    prediction_list = []

    total_gt = 0

    for image_index, (
        preds,
        gts
    ) in enumerate(
        zip(
            predictions,
            ground_truths
        )
    ):

        preds = filter_predictions(preds)

        total_gt += len(gts)

        for pred in preds:

            prediction_list.append(
                (
                    pred[4],
                    image_index,
                    pred
                )
            )

    if total_gt == 0:
        return 0.0

    prediction_list.sort(
        key=lambda x: x[0],
        reverse=True
    )

    tp = np.zeros(
        len(prediction_list)
    )

    fp = np.zeros(
        len(prediction_list)
    )

    matched_gts = {
        i: set()
        for i in range(len(ground_truths))
    }

    for index, (
        score,
        image_index,
        pred
    ) in enumerate(
        prediction_list
    ):

        gts = ground_truths[image_index]

        best_iou = 0.0
        best_gt = -1

        for gt_index, gt in enumerate(gts):

            if gt_index in matched_gts[image_index]:
                continue

            iou = calculate_iou(
                pred[:4],
                gt
            )

            if iou > best_iou:

                best_iou = iou
                best_gt = gt_index

        if (
            best_gt >= 0
            and best_iou >= iou_threshold
        ):

            tp[index] = 1
            matched_gts[image_index].add(
                best_gt
            )

        else:

            fp[index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / total_gt

    precisions = (
        cumulative_tp /
        (
            cumulative_tp +
            cumulative_fp
        )
    )

    # Precision envelope
    for i in range(
        len(precisions) - 2,
        -1,
        -1
    ):

        precisions[i] = max(
            precisions[i],
            precisions[i + 1]
        )

    recall_points = np.concatenate(
        (
            [0.0],
            recalls,
            [1.0]
        )
    )

    precision_points = np.concatenate(
        (
            [1.0],
            precisions,
            [0.0]
        )
    )

    ap = 0.0

    for i in range(
        len(recall_points) - 1
    ):

        ap += (
            recall_points[i + 1]
            - recall_points[i]
        ) * precision_points[i + 1]

    return float(ap)


# ============================================================
# mAP50 AND mAP50-95
# ============================================================

def calculate_map(
    predictions,
    ground_truths
):

    map50 = calculate_ap(
        predictions,
        ground_truths,
        0.50
    )

    aps = []

    for iou in np.arange(
        0.50,
        0.96,
        0.05
    ):

        aps.append(
            calculate_ap(
                predictions,
                ground_truths,
                float(iou)
            )
        )

    map5095 = float(
        np.mean(aps)
    )

    return (
        map50,
        map5095
    )


# ============================================================
# WBF
# ============================================================

def create_wbf_predictions(
    yolo_predictions,
    rtdetr_predictions,
    yolo_weight,
    rtdetr_weight
):

    final_predictions = []

    for yolo_preds, rt_preds in zip(
        yolo_predictions,
        rtdetr_predictions
    ):

        yolo_boxes = [
            p[:4]
            for p in yolo_preds
        ]

        yolo_scores = [
            p[4]
            for p in yolo_preds
        ]

        yolo_labels = [
            p[5]
            for p in yolo_preds
        ]

        rt_boxes = [
            p[:4]
            for p in rt_preds
        ]

        rt_scores = [
            p[4]
            for p in rt_preds
        ]

        rt_labels = [
            p[5]
            for p in rt_preds
        ]

        if (
            len(yolo_boxes) == 0
            and len(rt_boxes) == 0
        ):

            final_predictions.append([])
            continue

        fused_boxes, fused_scores, fused_labels = (
            weighted_boxes_fusion(
                [
                    yolo_boxes,
                    rt_boxes
                ],
                [
                    yolo_scores,
                    rt_scores
                ],
                [
                    yolo_labels,
                    rt_labels
                ],
                weights=[
                    yolo_weight,
                    rtdetr_weight
                ],
                iou_thr=WBF_IOU,
                skip_box_thr=WBF_SKIP,
                conf_type="avg"
            )
        )

        image_predictions = []

        for box, score, label in zip(
            fused_boxes,
            fused_scores,
            fused_labels
        ):

            image_predictions.append([
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(score),
                int(label)
            ])

        final_predictions.append(
            image_predictions
        )

    return final_predictions


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    predictions,
    ground_truths
):

    p, r, f1, tp, fp, fn = evaluate_prf(
        predictions,
        ground_truths
    )

    map50, map5095 = calculate_map(
        predictions,
        ground_truths
    )

    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "map50": map50,
        "map50_95": map5095,
        "TP": tp,
        "FP": fp,
        "FN": fn
    }


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

def save_predictions(
    predictions,
    image_files,
    output_file
):

    data = {}

    for image_path, preds in zip(
        image_files,
        predictions
    ):

        data[image_path.name] = preds

    with open(
        output_file,
        "w"
    ) as f:

        json.dump(
            data,
            f,
            indent=2
        )


# ============================================================
# RUN ONE SPLIT
# ============================================================

def run_split(
    split_name,
    yolo11_weight,
    yolo12_weight,
    rtdetr_weight,
    dataset_root
):

    print()
    print("=" * 80)
    print(f"UNIFIED STANDARDIZED EVALUATION — {split_name}")
    print("=" * 80)

    image_files = get_images(
        dataset_root
    )

    print(
        f"Test images: {len(image_files)}"
    )

    label_dir = (
        dataset_root /
        "labels" /
        "test"
    )

    ground_truths = []

    for image_path in image_files:

        label_path = (
            label_dir /
            f"{image_path.stem}.txt"
        )

        ground_truths.append(
            load_ground_truth(
                label_path
            )
        )

    print()
    print("Loading models...")

    model_yolo11 = YOLO(
        str(yolo11_weight)
    )

    model_yolo12 = YOLO(
        str(yolo12_weight)
    )

    model_rtdetr = YOLO(
        str(rtdetr_weight)
    )

    print("Models loaded.")

    # --------------------------------------------------------
    # GENERATE RAW PREDICTIONS
    # --------------------------------------------------------

    yolo11_predictions = []
    yolo12_predictions = []
    rtdetr_predictions = []

    print()
    print("Generating raw predictions...")

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"Processing {index}/{len(image_files)}",
            end="\r"
        )

        y11 = get_model_predictions(
            model_yolo11,
            image_path
        )

        y12 = get_model_predictions(
            model_yolo12,
            image_path
        )

        rt = get_model_predictions(
            model_rtdetr,
            image_path
        )

        yolo11_predictions.append(y11)
        yolo12_predictions.append(y12)
        rtdetr_predictions.append(rt)

    print()
    print("Raw predictions generated.")

    # --------------------------------------------------------
    # SAVE RAW PREDICTIONS
    # --------------------------------------------------------

    split_output = (
        OUTPUT_ROOT /
        split_name
    )

    split_output.mkdir(
        parents=True,
        exist_ok=True
    )

    save_predictions(
        yolo11_predictions,
        image_files,
        split_output /
        "YOLOv11_raw_predictions.json"
    )

    save_predictions(
        yolo12_predictions,
        image_files,
        split_output /
        "YOLOv12_raw_predictions.json"
    )

    save_predictions(
        rtdetr_predictions,
        image_files,
        split_output /
        "RT_DETR_raw_predictions.json"
    )

    # --------------------------------------------------------
    # STANDALONE MODELS
    # --------------------------------------------------------

    y11_result = evaluate_model(
        yolo11_predictions,
        ground_truths
    )

    y12_result = evaluate_model(
        yolo12_predictions,
        ground_truths
    )

    rt_result = evaluate_model(
        rtdetr_predictions,
        ground_truths
    )

    # --------------------------------------------------------
    # WBF — YOLOv11 + RT-DETR
    # --------------------------------------------------------

    y11_wbf = create_wbf_predictions(
        yolo11_predictions,
        rtdetr_predictions,
        0.2 if split_name == "70_20_10" else 0.3,
        0.8 if split_name == "70_20_10" else 0.7
    )

    y11_wbf_result = evaluate_model(
        y11_wbf,
        ground_truths
    )

    save_predictions(
        y11_wbf,
        image_files,
        split_output /
        "YOLOv11_RTDETR_WBF_predictions.json"
    )

    # --------------------------------------------------------
    # WBF — YOLOv12 + RT-DETR
    # --------------------------------------------------------

    y12_wbf = create_wbf_predictions(
        yolo12_predictions,
        rtdetr_predictions,
        0.2,
        0.8
    )

    y12_wbf_result = evaluate_model(
        y12_wbf,
        ground_truths
    )

    save_predictions(
        y12_wbf,
        image_files,
        split_output /
        "YOLOv12_RTDETR_WBF_predictions.json"
    )

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    results = {
        "YOLOv11": y11_result,
        "YOLOv12": y12_result,
        "RT-DETR": rt_result,
        "YOLOv11 + RT-DETR WBF": y11_wbf_result,
        "YOLOv12 + RT-DETR WBF": y12_wbf_result
    }

    print()
    print("=" * 80)
    print(f"FINAL NORMALIZED RESULTS — {split_name}")
    print("=" * 80)

    print(
        f"{'Model':<30}"
        f"{'P':>9}"
        f"{'R':>9}"
        f"{'F1':>9}"
        f"{'mAP50':>10}"
        f"{'mAP50-95':>12}"
    )

    print("-" * 80)

    for name, result in results.items():

        print(
            f"{name:<30}"
            f"{result['precision']:>9.4f}"
            f"{result['recall']:>9.4f}"
            f"{result['f1']:>9.4f}"
            f"{result['map50']:>10.4f}"
            f"{result['map50_95']:>12.4f}"
        )

    # --------------------------------------------------------
    # WBF LIFT
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print(f"WBF PERFORMANCE LIFT — {split_name}")
    print("=" * 80)

    comparisons = [
        (
            "YOLOv11 + RT-DETR WBF",
            "YOLOv11",
            y11_wbf_result
        ),
        (
            "YOLOv12 + RT-DETR WBF",
            "YOLOv12",
            y12_wbf_result
        )
    ]

    for ensemble_name, baseline_name, ensemble in comparisons:

        baseline = results[baseline_name]

        print()
        print(
            f"{ensemble_name} vs {baseline_name}"
        )

        for metric in [
            "precision",
            "recall",
            "f1",
            "map50",
            "map50_95"
        ]:

            change = (
                ensemble[metric]
                - baseline[metric]
            )

            print(
                f"{metric:<12}: "
                f"{baseline[metric]:.4f}"
                f" -> "
                f"{ensemble[metric]:.4f}"
                f"  "
                f"({'+' if change >= 0 else ''}"
                f"{change:.4f})"
            )

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    output_txt = (
        split_output /
        "UNIFIED_FINAL_RESULTS.txt"
    )

    with open(
        output_txt,
        "w"
    ) as f:

        f.write(
            f"UNIFIED STANDARDIZED EVALUATION\n"
        )

        f.write(
            f"Split: {split_name}\n\n"
        )

        f.write(
            "Settings:\n"
        )

        f.write(
            f"Prediction confidence: {PRED_CONF}\n"
        )

        f.write(
            f"Evaluation confidence: {EVAL_CONF}\n"
        )

        f.write(
            f"Evaluation IoU: {EVAL_IOU}\n"
        )

        f.write(
            f"WBF IoU: {WBF_IOU}\n"
        )

        f.write(
            f"WBF skip threshold: {WBF_SKIP}\n\n"
        )

        for name, result in results.items():

            f.write(
                f"{name}\n"
            )

            f.write(
                f"Precision: {result['precision']:.6f}\n"
            )

            f.write(
                f"Recall: {result['recall']:.6f}\n"
            )

            f.write(
                f"F1: {result['f1']:.6f}\n"
            )

            f.write(
                f"mAP50: {result['map50']:.6f}\n"
            )

            f.write(
                f"mAP50-95: {result['map50_95']:.6f}\n"
            )

            f.write(
                f"TP: {result['TP']}\n"
            )

            f.write(
                f"FP: {result['FP']}\n"
            )

            f.write(
                f"FN: {result['FN']}\n\n"
            )

    print()
    print(
        f"Saved results to:\n{output_txt}"
    )

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    all_results = {}

    for split_name, config in DATASETS.items():

        # Verify paths
        for key in [
            "root",
            "yolo11",
            "yolo12",
            "rtdetr"
        ]:

            if not config[key].exists():

                raise FileNotFoundError(
                    f"Missing path:\n{config[key]}"
                )

        all_results[split_name] = run_split(
            split_name,
            config["yolo11"],
            config["yolo12"],
            config["rtdetr"],
            config["root"]
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print("=" * 100)
    print("COMPLETE UNIFIED STANDARDIZED COMPARISON")
    print("=" * 100)

    for split_name, results in all_results.items():

        print()
        print(split_name)

        print("-" * 100)

        for name, result in results.items():

            print(
                f"{name:<30}"
                f"P={result['precision']:.4f}  "
                f"R={result['recall']:.4f}  "
                f"F1={result['f1']:.4f}  "
                f"mAP50={result['map50']:.4f}  "
                f"mAP50-95={result['map50_95']:.4f}"
            )

    print()
    print("=" * 100)
    print("UNIFIED EVALUATION COMPLETED")
    print("=" * 100)