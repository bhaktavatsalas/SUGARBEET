from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# YOLOv11 + RT-DETR WEIGHTED BOX FUSION
# Dataset: 70:20:10
# Validation-based optimization
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
# LOAD DATASET
# ============================================================

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

DATASET_PATH = Path(data["path"])

VAL_IMAGES = DATASET_PATH / data["val"]
VAL_LABELS = DATASET_PATH / "labels/val"


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLOv11...")
yolo_model = YOLO(str(YOLOV11_MODEL))

print("Loading RT-DETR...")
rtdetr_model = YOLO(str(RTDETR_MODEL))

print("\nModels loaded successfully.")


# ============================================================
# GET MODEL PREDICTIONS
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

        # Normalize
        x1 = x1 / w
        x2 = x2 / w
        y1 = y1 / h
        y2 = y2 / h

        # Explicitly clip to [0,1]
        x1 = np.clip(x1, 0.0, 1.0)
        y1 = np.clip(y1, 0.0, 1.0)
        x2 = np.clip(x2, 0.0, 1.0)
        y2 = np.clip(y2, 0.0, 1.0)

        # Make sure coordinates are ordered correctly
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
    labels = []

    if not label_path.exists():
        return boxes, labels

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

            labels.append(int(cls))

    return boxes, labels


# ============================================================
# WBF
# ============================================================

def fuse_predictions(
    yolo_boxes,
    yolo_scores,
    yolo_labels,
    rt_boxes,
    rt_scores,
    rt_labels,
    yolo_weight,
    rt_weight
):

    boxes_list = [
        yolo_boxes,
        rt_boxes
    ]

    scores_list = [
        yolo_scores,
        rt_scores
    ]

    labels_list = [
        yolo_labels,
        rt_labels
    ]

    fused_boxes, fused_scores, fused_labels = (
        weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=[
                yolo_weight,
                rt_weight
            ],
            iou_thr=0.55,
            skip_box_thr=0.05,
            conf_type="avg"
        )
    )

    # Explicitly clip final boxes
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

    intersection = max(
        0,
        x2 - x1
    ) * max(
        0,
        y2 - y1
    )

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
# MATCH PREDICTIONS
# ============================================================

def get_matches(
    pred_boxes,
    pred_scores,
    gt_boxes,
    iou_threshold
):

    if len(pred_boxes) == 0:

        return (
            np.array([], dtype=bool),
            0
        )

    if len(gt_boxes) == 0:

        return (
            np.zeros(
                len(pred_boxes),
                dtype=bool
            ),
            0
        )

    order = np.argsort(
        np.asarray(pred_scores)
    )[::-1]

    sorted_boxes = [
        pred_boxes[i]
        for i in order
    ]

    matched_gt = set()

    tp = np.zeros(
        len(sorted_boxes),
        dtype=bool
    )

    for i, pred_box in enumerate(sorted_boxes):

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

            tp[i] = True
            matched_gt.add(best_gt)

    return tp, len(matched_gt)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def evaluate_at_threshold(
    predictions,
    confidence_threshold,
    iou_threshold=0.50
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for item in predictions:

        filtered_boxes = []
        filtered_scores = []

        for box, score in zip(
            item["boxes"],
            item["scores"]
        ):

            if score >= confidence_threshold:

                filtered_boxes.append(box)
                filtered_scores.append(score)

        tp_flags, matched = get_matches(
            filtered_boxes,
            filtered_scores,
            item["gt_boxes"],
            iou_threshold
        )

        tp = int(np.sum(tp_flags))

        fp = len(filtered_boxes) - tp

        fn = len(item["gt_boxes"]) - matched

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

    return precision, recall, f1


# ============================================================
# AP CALCULATION
# ============================================================

def calculate_ap(
    predictions,
    confidence_threshold,
    iou_threshold
):

    all_predictions = []

    total_gt = 0

    for item in predictions:

        total_gt += len(
            item["gt_boxes"]
        )

        for box, score in zip(
            item["boxes"],
            item["scores"]
        ):

            if score >= confidence_threshold:

                all_predictions.append({
                    "box": box,
                    "score": float(score),
                    "gt_boxes": item["gt_boxes"],
                    "image_id": item["image_id"]
                })

    if total_gt == 0:
        return 0.0

    if len(all_predictions) == 0:
        return 0.0

    # Sort globally by confidence
    all_predictions.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = {}

    tp = np.zeros(
        len(all_predictions),
        dtype=np.float64
    )

    fp = np.zeros(
        len(all_predictions),
        dtype=np.float64
    )

    for i, pred in enumerate(all_predictions):

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

    recalls = (
        cumulative_tp /
        total_gt
    )

    precisions = (
        cumulative_tp /
        (
            cumulative_tp +
            cumulative_fp +
            1e-16
        )
    )

    # Precision envelope
    mrec = np.concatenate(
        ([0.0], recalls, [1.0])
    )

    mpre = np.concatenate(
        ([0.0], precisions, [0.0])
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

    # 101-point interpolation
    recall_points = np.linspace(
        0,
        1,
        101
    )

    precision_points = []

    for recall_point in recall_points:

        indices = np.where(
            mrec >= recall_point
        )[0]

        if len(indices) > 0:

            precision_points.append(
                np.max(
                    mpre[indices]
                )
            )

        else:

            precision_points.append(
                0.0
            )

    ap = np.mean(
        precision_points
    )

    return float(ap)


# ============================================================
# mAP50-95
# ============================================================

def calculate_map(
    predictions,
    confidence_threshold
):

    aps = []

    for iou_threshold in np.arange(
        0.50,
        0.96,
        0.05
    ):

        ap = calculate_ap(
            predictions,
            confidence_threshold,
            float(iou_threshold)
        )

        aps.append(ap)

    map50 = aps[0]

    map50_95 = np.mean(
        aps
    )

    return map50, map50_95


# ============================================================
# GENERATE VALIDATION PREDICTIONS
# ============================================================

def prepare_validation_predictions():

    print("\n" + "=" * 60)
    print(
        "GENERATING VALIDATION PREDICTIONS"
    )
    print("=" * 60)

    image_files = sorted(
        list(
            VAL_IMAGES.glob("*.jpg")
        )
        + list(
            VAL_IMAGES.glob("*.jpeg")
        )
        + list(
            VAL_IMAGES.glob("*.png")
        )
    )

    print(
        f"Validation images: "
        f"{len(image_files)}"
    )

    predictions = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\rProcessing validation image "
            f"{index}/{len(image_files)}",
            end=""
        )

        label_path = (
            VAL_LABELS /
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

        gt_boxes, gt_labels = (
            read_ground_truth(
                label_path
            )
        )

        predictions.append({

            "image_id": image_path.stem,

            "yolo_boxes": yolo_boxes,
            "yolo_scores": yolo_scores,
            "yolo_labels": yolo_labels,

            "rt_boxes": rt_boxes,
            "rt_scores": rt_scores,
            "rt_labels": rt_labels,

            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels
        })

    print(
        "\nValidation predictions generated."
    )

    return predictions


# ============================================================
# TEST WBF WEIGHT COMBINATIONS
# ============================================================

def optimize_weights(
    validation_predictions
):

    print("\n" + "=" * 60)
    print(
        "OPTIMIZING WBF WEIGHTS"
    )
    print("=" * 60)

    candidate_weights = [

        (0.5, 0.5),

        (0.4, 0.6),

        (0.3, 0.7),

        (0.2, 0.8),

        (0.6, 0.4),

        (0.7, 0.3),

        (0.8, 0.2)
    ]

    confidence_thresholds = [
        0.25,
        0.35,
        0.45,
        0.55
    ]

    results = []

    for yolo_weight, rt_weight in (
        candidate_weights
    ):

        print(
            "\n"
            + "-" * 60
        )

        print(
            f"YOLOv11 weight = "
            f"{yolo_weight}"
        )

        print(
            f"RT-DETR weight = "
            f"{rt_weight}"
        )

        fused_predictions = []

        for item in validation_predictions:

            boxes, scores, labels = (
                fuse_predictions(
                    item["yolo_boxes"],
                    item["yolo_scores"],
                    item["yolo_labels"],
                    item["rt_boxes"],
                    item["rt_scores"],
                    item["rt_labels"],
                    yolo_weight,
                    rt_weight
                )
            )

            fused_predictions.append({

                "image_id":
                    item["image_id"],

                "boxes":
                    boxes,

                "scores":
                    scores,

                "labels":
                    labels,

                "gt_boxes":
                    item["gt_boxes"]
            })

        for confidence_threshold in (
            confidence_thresholds
        ):

            precision, recall, f1 = (
                evaluate_at_threshold(
                    fused_predictions,
                    confidence_threshold,
                    0.50
                )
            )

            map50, map50_95 = (
                calculate_map(
                    fused_predictions,
                    confidence_threshold
                )
            )

            print(
                f"Conf={confidence_threshold:.2f} | "
                f"P={precision:.4f} | "
                f"R={recall:.4f} | "
                f"F1={f1:.4f} | "
                f"mAP50={map50:.4f} | "
                f"mAP50-95={map50_95:.4f}"
            )

            results.append({

                "yolo_weight":
                    yolo_weight,

                "rt_weight":
                    rt_weight,

                "confidence":
                    confidence_threshold,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,

                "map50":
                    map50,

                "map50_95":
                    map50_95
            })

    # Best configuration based on validation mAP50-95
    best = max(
        results,
        key=lambda x: x["map50_95"]
    )

    print("\n" + "=" * 60)
    print(
        "BEST WBF CONFIGURATION"
    )
    print("=" * 60)

    print(
        f"YOLOv11 weight  : "
        f"{best['yolo_weight']}"
    )

    print(
        f"RT-DETR weight  : "
        f"{best['rt_weight']}"
    )

    print(
        f"Confidence      : "
        f"{best['confidence']}"
    )

    print(
        f"Precision       : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall          : "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1              : "
        f"{best['f1']:.4f}"
    )

    print(
        f"mAP50           : "
        f"{best['map50']:.4f}"
    )

    print(
        f"mAP50-95        : "
        f"{best['map50_95']:.4f}"
    )

    return best, results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    best,
    results
):

    output_file = (
        OUTPUT_DIR /
        "wbf_validation_results.txt"
    )

    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            "YOLOv11 + RT-DETR "
            "Weighted Box Fusion\n"
        )

        f.write(
            "Dataset Split: 70:20:10\n\n"
        )

        f.write(
            "BEST CONFIGURATION\n"
        )

        f.write(
            f"YOLOv11 Weight: "
            f"{best['yolo_weight']}\n"
        )

        f.write(
            f"RT-DETR Weight: "
            f"{best['rt_weight']}\n"
        )

        f.write(
            f"Confidence Threshold: "
            f"{best['confidence']}\n"
        )

        f.write(
            f"Precision: "
            f"{best['precision']:.6f}\n"
        )

        f.write(
            f"Recall: "
            f"{best['recall']:.6f}\n"
        )

        f.write(
            f"F1: "
            f"{best['f1']:.6f}\n"
        )

        f.write(
            f"mAP50: "
            f"{best['map50']:.6f}\n"
        )

        f.write(
            f"mAP50-95: "
            f"{best['map50_95']:.6f}\n"
        )

        f.write(
            "\n\nALL VALIDATION RESULTS\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        for result in results:

            f.write(
                f"YOLOv11="
                f"{result['yolo_weight']} | "
                f"RT-DETR="
                f"{result['rt_weight']} | "
                f"Conf="
                f"{result['confidence']:.2f} | "
                f"P="
                f"{result['precision']:.6f} | "
                f"R="
                f"{result['recall']:.6f} | "
                f"F1="
                f"{result['f1']:.6f} | "
                f"mAP50="
                f"{result['map50']:.6f} | "
                f"mAP50-95="
                f"{result['map50_95']:.6f}\n"
            )

    print(
        "\nResults saved to:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    validation_predictions = (
        prepare_validation_predictions()
    )

    best, results = (
        optimize_weights(
            validation_predictions
        )
    )

    save_results(
        best,
        results
    )

    print("\n" + "=" * 60)
    print(
        "WBF VALIDATION COMPLETED"
    )
    print("=" * 60)

    print(
        "\nIMPORTANT:"
    )

    print(
        "The selected weights are based ONLY "
        "on the validation set."
    )

    print(
        "Do NOT use the test set to select "
        "weights or confidence."
    )

    print(
        "\nNext step: evaluate the selected "
        "configuration on the 70:20:10 TEST set."
    )