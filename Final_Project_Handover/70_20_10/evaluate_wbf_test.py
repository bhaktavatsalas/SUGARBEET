from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# YOLOv11 + RT-DETR WBF — FINAL TEST EVALUATION
# Dataset: 70:20:10
#
# LOCKED from validation:
# YOLOv11 weight = 0.2
# RT-DETR weight = 0.8
# Confidence threshold = 0.25
# ============================================================

BASE = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/70_20_10"
)

DATA_YAML = BASE / "sugarbeet_70_20_10.yaml"

YOLOV11_MODEL = (
    BASE / "YOLOv11_70_20_10_Detection/weights/best.pt"
)

RTDETR_MODEL = (
    BASE / "RT_DETR_70_20_10/weights/best.pt"
)

OUTPUT_DIR = BASE / "YOLOv11_RTDETR_WBF_70_20_10"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOCKED VALIDATION CONFIGURATION
# ============================================================

YOLO_WEIGHT = 0.2
RT_WEIGHT = 0.8
CONF_THRESHOLD = 0.25
WBF_IOU_THRESHOLD = 0.55


# ============================================================
# LOAD DATASET
# ============================================================

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

DATASET_PATH = Path(data["path"])

TEST_IMAGES = DATASET_PATH / data["test"]
TEST_LABELS = DATASET_PATH / "labels/test"


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLOv11...")
yolo_model = YOLO(str(YOLOV11_MODEL))

print("Loading RT-DETR...")
rtdetr_model = YOLO(str(RTDETR_MODEL))

print("\nModels loaded successfully.")


# ============================================================
# GET PREDICTIONS
# ============================================================

def get_predictions(model, image_path):

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        verbose=False
    )

    result = results[0]

    if result.boxes is None or len(result.boxes) == 0:
        return [], [], []

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    labels = result.boxes.cls.cpu().numpy().astype(int)

    h, w = result.orig_shape

    normalized_boxes = []

    for box in boxes:

        x1, y1, x2, y2 = box

        x1 = np.clip(x1 / w, 0.0, 1.0)
        y1 = np.clip(y1 / h, 0.0, 1.0)
        x2 = np.clip(x2 / w, 0.0, 1.0)
        y2 = np.clip(y2 / h, 0.0, 1.0)

        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)

        normalized_boxes.append([
            float(x1),
            float(y1),
            float(x2),
            float(y2)
        ])

    return (
        normalized_boxes,
        scores.tolist(),
        labels.tolist()
    )


# ============================================================
# READ GROUND TRUTH
# ============================================================

def read_ground_truth(label_path):

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
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ])

    return boxes


# ============================================================
# WBF
# ============================================================

def fuse_predictions(
    yolo_boxes,
    yolo_scores,
    yolo_labels,
    rt_boxes,
    rt_scores,
    rt_labels
):

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
                YOLO_WEIGHT,
                RT_WEIGHT
            ],
            iou_thr=WBF_IOU_THRESHOLD,
            skip_box_thr=0.05,
            conf_type="avg"
        )
    )

    fused_boxes = np.clip(
        np.asarray(fused_boxes),
        0.0,
        1.0
    )

    return (
        fused_boxes.tolist(),
        fused_scores.tolist(),
        fused_labels.astype(int).tolist()
    )


# ============================================================
# IoU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# MATCH PREDICTIONS
# ============================================================

def match_predictions(
    pred_boxes,
    pred_scores,
    gt_boxes,
    iou_threshold
):

    if len(pred_boxes) == 0:
        return 0, 0, len(gt_boxes)

    if len(gt_boxes) == 0:
        return 0, len(pred_boxes), 0

    order = np.argsort(
        np.asarray(pred_scores)
    )[::-1]

    sorted_boxes = [
        pred_boxes[i]
        for i in order
    ]

    matched_gt = set()

    tp = 0
    fp = 0

    for pred_box in sorted_boxes:

        best_iou = 0.0
        best_gt = -1

        for j, gt_box in enumerate(gt_boxes):

            if j in matched_gt:
                continue

            iou = calculate_iou(
                pred_box,
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if (
            best_gt >= 0
            and best_iou >= iou_threshold
        ):

            tp += 1
            matched_gt.add(best_gt)

        else:

            fp += 1

    fn = len(gt_boxes) - len(matched_gt)

    return tp, fp, fn


# ============================================================
# EVALUATE AT IoU
# ============================================================

def evaluate_at_iou(
    predictions,
    iou_threshold
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for item in predictions:

        boxes = []
        scores = []

        for box, score in zip(
            item["boxes"],
            item["scores"]
        ):

            if score >= CONF_THRESHOLD:

                boxes.append(box)
                scores.append(score)

        tp, fp, fn = match_predictions(
            boxes,
            scores,
            item["gt_boxes"],
            iou_threshold
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = (
        total_tp /
        (total_tp + total_fp)
        if total_tp + total_fp > 0
        else 0.0
    )

    recall = (
        total_tp /
        (total_tp + total_fn)
        if total_tp + total_fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return (
        precision,
        recall,
        f1,
        total_tp,
        total_fp,
        total_fn
    )


# ============================================================
# AP CALCULATION
# ============================================================

def calculate_ap(
    predictions,
    iou_threshold
):

    all_preds = []
    total_gt = 0

    for item in predictions:

        total_gt += len(
            item["gt_boxes"]
        )

        for box, score in zip(
            item["boxes"],
            item["scores"]
        ):

            if score >= CONF_THRESHOLD:

                all_preds.append({
                    "box": box,
                    "score": float(score),
                    "gt_boxes": item["gt_boxes"],
                    "image_id": item["image_id"]
                })

    if total_gt == 0 or len(all_preds) == 0:
        return 0.0

    all_preds.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = {}

    tp = np.zeros(
        len(all_preds)
    )

    fp = np.zeros(
        len(all_preds)
    )

    for i, pred in enumerate(all_preds):

        image_id = pred["image_id"]

        if image_id not in matched:
            matched[image_id] = set()

        best_iou = 0.0
        best_gt = -1

        for j, gt_box in enumerate(
            pred["gt_boxes"]
        ):

            if j in matched[image_id]:
                continue

            iou = calculate_iou(
                pred["box"],
                gt_box
            )

            if iou > best_iou:
                best_iou = iou
                best_gt = j

        if (
            best_gt >= 0
            and best_iou >= iou_threshold
        ):

            tp[i] = 1
            matched[image_id].add(
                best_gt
            )

        else:

            fp[i] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recall = (
        cumulative_tp /
        total_gt
    )

    precision = (
        cumulative_tp /
        (
            cumulative_tp
            + cumulative_fp
            + 1e-16
        )
    )

    mrec = np.concatenate(
        ([0.0], recall, [1.0])
    )

    mpre = np.concatenate(
        ([0.0], precision, [0.0])
    )

    for i in range(
        len(mpre) - 1,
        0,
        -1
    ):

        mpre[i - 1] = max(
            mpre[i - 1],
            mpre[i]
        )

    recall_points = np.linspace(
        0,
        1,
        101
    )

    ap_values = []

    for recall_point in recall_points:

        indices = np.where(
            mrec >= recall_point
        )[0]

        if len(indices) > 0:
            ap_values.append(
                np.max(
                    mpre[indices]
                )
            )
        else:
            ap_values.append(0.0)

    return float(
        np.mean(ap_values)
    )


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

def generate_test_predictions():

    print("\n" + "=" * 60)
    print(
        "GENERATING FINAL TEST PREDICTIONS"
    )
    print("=" * 60)

    image_files = sorted(
        list(TEST_IMAGES.glob("*.jpg"))
        + list(TEST_IMAGES.glob("*.jpeg"))
        + list(TEST_IMAGES.glob("*.png"))
    )

    print(
        f"Test images: "
        f"{len(image_files)}"
    )

    predictions = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\rProcessing test image "
            f"{index}/{len(image_files)}",
            end=""
        )

        label_path = (
            TEST_LABELS /
            f"{image_path.stem}.txt"
        )

        yolo_boxes, yolo_scores, yolo_labels = (
            get_predictions(
                yolo_model,
                image_path
            )
        )

        rt_boxes, rt_scores, rt_labels = (
            get_predictions(
                rtdetr_model,
                image_path
            )
        )

        fused_boxes, fused_scores, fused_labels = (
            fuse_predictions(
                yolo_boxes,
                yolo_scores,
                yolo_labels,
                rt_boxes,
                rt_scores,
                rt_labels
            )
        )

        gt_boxes = read_ground_truth(
            label_path
        )

        predictions.append({

            "image_id": image_path.stem,

            "boxes": fused_boxes,

            "scores": fused_scores,

            "labels": fused_labels,

            "gt_boxes": gt_boxes
        })

    print(
        "\nTest predictions generated."
    )

    return predictions


# ============================================================
# MAIN EVALUATION
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print(
        "YOLOv11 + RT-DETR WBF"
    )
    print(
        "FINAL 70:20:10 TEST EVALUATION"
    )
    print("=" * 60)

    print(
        f"\nYOLOv11 weight : {YOLO_WEIGHT}"
    )

    print(
        f"RT-DETR weight : {RT_WEIGHT}"
    )

    print(
        f"Confidence     : {CONF_THRESHOLD}"
    )

    print(
        f"WBF IoU        : {WBF_IOU_THRESHOLD}"
    )

    print(
        "\nThese parameters were locked "
        "from validation."
    )

    predictions = (
        generate_test_predictions()
    )

    # --------------------------------------------------------
    # Standard detection metrics
    # --------------------------------------------------------

    (
        precision,
        recall,
        f1,
        tp,
        fp,
        fn
    ) = evaluate_at_iou(
        predictions,
        0.50
    )

    # --------------------------------------------------------
    # mAP
    # --------------------------------------------------------

    aps = []

    for iou in np.arange(
        0.50,
        0.96,
        0.05
    ):

        ap = calculate_ap(
            predictions,
            float(iou)
        )

        aps.append(ap)

    map50 = aps[0]
    map50_95 = np.mean(aps)

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "FINAL WBF TEST RESULTS"
    )
    print("=" * 60)

    print(
        f"\nPrecision       : {precision:.4f}"
    )

    print(
        f"Recall          : {recall:.4f}"
    )

    print(
        f"F1 Score        : {f1:.4f}"
    )

    print(
        f"mAP@50          : {map50:.4f}"
    )

    print(
        f"mAP@50-95       : {map50_95:.4f}"
    )

    print(
        f"\nTP              : {tp}"
    )

    print(
        f"FP              : {fp}"
    )

    print(
        f"FN              : {fn}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR /
        "wbf_test_results.txt"
    )

    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            "YOLOv11 + RT-DETR Weighted Box Fusion\n"
        )

        f.write(
            "Dataset Split: 70:20:10\n"
        )

        f.write(
            "Evaluation Set: TEST\n\n"
        )

        f.write(
            "LOCKED VALIDATION CONFIGURATION\n"
        )

        f.write(
            f"YOLOv11 Weight: "
            f"{YOLO_WEIGHT}\n"
        )

        f.write(
            f"RT-DETR Weight: "
            f"{RT_WEIGHT}\n"
        )

        f.write(
            f"Confidence Threshold: "
            f"{CONF_THRESHOLD}\n"
        )

        f.write(
            f"WBF IoU Threshold: "
            f"{WBF_IOU_THRESHOLD}\n\n"
        )

        f.write(
            "FINAL TEST METRICS\n"
        )

        f.write(
            f"Precision: "
            f"{precision:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{recall:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{f1:.6f}\n"
        )

        f.write(
            f"mAP50: "
            f"{map50:.6f}\n"
        )

        f.write(
            f"mAP50-95: "
            f"{map50_95:.6f}\n"
        )

        f.write(
            f"TP: {tp}\n"
        )

        f.write(
            f"FP: {fp}\n"
        )

        f.write(
            f"FN: {fn}\n"
        )

    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )

    print("\n" + "=" * 60)
    print(
        "FINAL TEST EVALUATION COMPLETED"
    )
    print("=" * 60)