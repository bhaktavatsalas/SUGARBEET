from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# YOLOv11 + RT-DETR WBF INTEGRATION
# Dataset: 80:10:10
#
# Validation is used ONLY to select:
#   1. WBF model weights
#   2. Confidence threshold
#
# Test set remains untouched until the final evaluation.
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/80_10_10"
)

DATA_YAML = (
    BASE.parent.parent
    / "sugarbeet_80_10_10_detection.yaml"
)

YOLOV11_MODEL = (
    BASE / "YOLOv11/detection/weights/best.pt"
)

RTDETR_MODEL = (
    BASE / "RT_DETR4/weights/best.pt"
)

OUTPUT_DIR = (
    BASE / "YOLOv11_RTDETR_WBF_80_10_10"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

DATASET_PATH = Path(data["path"])

VAL_IMAGES = (
    DATASET_PATH / data["val"]
)

VAL_LABELS = (
    DATASET_PATH / "labels/val"
)


# ============================================================
# CHECK PATHS
# ============================================================

print("\n" + "=" * 65)
print("80:10:10 YOLOv11 + RT-DETR WBF")
print("=" * 65)

print("\nDataset YAML:")
print(DATA_YAML)

print("\nDataset:")
print(DATASET_PATH)

print("\nYOLOv11:")
print(YOLOV11_MODEL)

print("\nRT-DETR:")
print(RTDETR_MODEL)

if not DATA_YAML.exists():
    raise FileNotFoundError(
        f"Dataset YAML not found:\n{DATA_YAML}"
    )

if not YOLOV11_MODEL.exists():
    raise FileNotFoundError(
        f"YOLOv11 best.pt not found:\n{YOLOV11_MODEL}"
    )

if not RTDETR_MODEL.exists():
    raise FileNotFoundError(
        f"RT-DETR best.pt not found:\n{RTDETR_MODEL}"
    )

if not VAL_IMAGES.exists():
    raise FileNotFoundError(
        f"Validation images not found:\n{VAL_IMAGES}"
    )

if not VAL_LABELS.exists():
    raise FileNotFoundError(
        f"Validation labels not found:\n{VAL_LABELS}"
    )

print("\nAll required paths verified.")


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading YOLOv11...")
yolo_model = YOLO(
    str(YOLOV11_MODEL)
)

print("Loading RT-DETR...")
rtdetr_model = YOLO(
    str(RTDETR_MODEL)
)

print("\nModels loaded successfully.")


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_predictions(
    model,
    image_path
):

    results = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=0.001,
        iou=0.7,
        verbose=False
    )

    result = results[0]

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):
        return [], [], []

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    scores = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    labels = (
        result.boxes.cls
        .cpu()
        .numpy()
        .astype(int)
    )

    h, w = result.orig_shape

    normalized_boxes = []

    for box in boxes:

        x1, y1, x2, y2 = box

        x1 = np.clip(
            x1 / w,
            0.0,
            1.0
        )

        y1 = np.clip(
            y1 / h,
            0.0,
            1.0
        )

        x2 = np.clip(
            x2 / w,
            0.0,
            1.0
        )

        y2 = np.clip(
            y2 / h,
            0.0,
            1.0
        )

        x1, x2 = (
            min(x1, x2),
            max(x1, x2)
        )

        y1, y2 = (
            min(y1, y2),
            max(y1, y2)
        )

        normalized_boxes.append(
            [
                float(x1),
                float(y1),
                float(x2),
                float(y2)
            ]
        )

    return (
        normalized_boxes,
        scores.tolist(),
        labels.tolist()
    )


# ============================================================
# GROUND TRUTH
# ============================================================

def read_ground_truth(
    label_path
):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(
        label_path,
        "r"
    ) as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            cls, xc, yc, bw, bh = (
                map(float, parts)
            )

            x1 = xc - bw / 2
            y1 = yc - bh / 2

            x2 = xc + bw / 2
            y2 = yc + bh / 2

            boxes.append(
                [
                    float(x1),
                    float(y1),
                    float(x2),
                    float(y2)
                ]
            )

    return boxes


# ============================================================
# WEIGHTED BOX FUSION
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
                rt_weight
            ],
            iou_thr=0.55,
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

def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection = (
        max(0, x2 - x1)
        *
        max(0, y2 - y1)
    )

    area1 = (
        max(0, box1[2] - box1[0])
        *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0])
        *
        max(0, box2[3] - box2[1])
    )

    union = (
        area1
        + area2
        - intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

def evaluate_prf(
    predictions,
    confidence_threshold,
    iou_threshold=0.50
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for item in predictions:

        filtered = []

        for box, score in zip(
            item["boxes"],
            item["scores"]
        ):

            if score >= confidence_threshold:

                filtered.append(
                    (
                        box,
                        score
                    )
                )

        filtered.sort(
            key=lambda x: x[1],
            reverse=True
        )

        matched_gt = set()

        for pred_box, pred_score in filtered:

            best_iou = 0.0
            best_gt = -1

            for j, gt_box in enumerate(
                item["gt_boxes"]
            ):

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

                total_tp += 1

                matched_gt.add(
                    best_gt
                )

            else:

                total_fp += 1

        total_fn += (
            len(item["gt_boxes"])
            - len(matched_gt)
        )

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
        f1
    )


# ============================================================
# AP
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

                all_predictions.append(
                    {
                        "image_id":
                            item["image_id"],

                        "box":
                            box,

                        "score":
                            float(score),

                        "gt_boxes":
                            item["gt_boxes"]
                    }
                )

    if (
        total_gt == 0
        or len(all_predictions) == 0
    ):
        return 0.0

    all_predictions.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    matched = {}

    tp = np.zeros(
        len(all_predictions)
    )

    fp = np.zeros(
        len(all_predictions)
    )

    for i, prediction in enumerate(
        all_predictions
    ):

        image_id = (
            prediction["image_id"]
        )

        if image_id not in matched:
            matched[image_id] = set()

        best_iou = 0.0
        best_gt = -1

        for j, gt_box in enumerate(
            prediction["gt_boxes"]
        ):

            if j in matched[image_id]:
                continue

            iou = calculate_iou(
                prediction["box"],
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

            matched[
                image_id
            ].add(best_gt)

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
        (
            [0.0],
            recall,
            [1.0]
        )
    )

    mpre = np.concatenate(
        (
            [0.0],
            precision,
            [0.0]
        )
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

            ap_values.append(
                0.0
            )

    return float(
        np.mean(ap_values)
    )


# ============================================================
# mAP
# ============================================================

def calculate_map(
    predictions,
    confidence_threshold
):

    aps = []

    for iou in np.arange(
        0.50,
        0.96,
        0.05
    ):

        ap = calculate_ap(
            predictions,
            confidence_threshold,
            float(iou)
        )

        aps.append(ap)

    map50 = aps[0]

    map50_95 = np.mean(
        aps
    )

    return (
        map50,
        map50_95
    )


# ============================================================
# GENERATE VALIDATION PREDICTIONS
# ============================================================

def generate_validation_predictions():

    print("\n" + "=" * 65)
    print(
        "GENERATING VALIDATION PREDICTIONS"
    )
    print("=" * 65)

    image_files = sorted(
        list(
            VAL_IMAGES.glob("*.jpg")
        )
        +
        list(
            VAL_IMAGES.glob("*.jpeg")
        )
        +
        list(
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
            f"\rProcessing validation "
            f"image {index}/"
            f"{len(image_files)}",
            end=""
        )

        label_path = (
            VAL_LABELS
            /
            f"{image_path.stem}.txt"
        )

        (
            yolo_boxes,
            yolo_scores,
            yolo_labels
        ) = get_predictions(
            yolo_model,
            image_path
        )

        (
            rt_boxes,
            rt_scores,
            rt_labels
        ) = get_predictions(
            rtdetr_model,
            image_path
        )

        gt_boxes = read_ground_truth(
            label_path
        )

        predictions.append(
            {
                "image_id":
                    image_path.stem,

                "yolo_boxes":
                    yolo_boxes,

                "yolo_scores":
                    yolo_scores,

                "yolo_labels":
                    yolo_labels,

                "rt_boxes":
                    rt_boxes,

                "rt_scores":
                    rt_scores,

                "rt_labels":
                    rt_labels,

                "gt_boxes":
                    gt_boxes
            }
        )

    print(
        "\nValidation predictions generated."
    )

    return predictions


# ============================================================
# OPTIMIZE WBF
# ============================================================

def optimize_wbf(
    validation_predictions
):

    print("\n" + "=" * 65)
    print(
        "OPTIMIZING WBF ON VALIDATION SET"
    )
    print("=" * 65)

    candidate_weights = [

        (0.2, 0.8),
        (0.3, 0.7),
        (0.4, 0.6),
        (0.5, 0.5),
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

        print("\n" + "-" * 65)

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

            (
                boxes,
                scores,
                labels
            ) = fuse_predictions(
                item["yolo_boxes"],
                item["yolo_scores"],
                item["yolo_labels"],
                item["rt_boxes"],
                item["rt_scores"],
                item["rt_labels"],
                yolo_weight,
                rt_weight
            )

            fused_predictions.append(
                {
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
                }
            )

        for confidence in (
            confidence_thresholds
        ):

            (
                precision,
                recall,
                f1
            ) = evaluate_prf(
                fused_predictions,
                confidence,
                0.50
            )

            (
                map50,
                map50_95
            ) = calculate_map(
                fused_predictions,
                confidence
            )

            print(
                f"Conf={confidence:.2f} | "
                f"P={precision:.4f} | "
                f"R={recall:.4f} | "
                f"F1={f1:.4f} | "
                f"mAP50={map50:.4f} | "
                f"mAP50-95={map50_95:.4f}"
            )

            results.append(
                {
                    "yolo_weight":
                        yolo_weight,

                    "rt_weight":
                        rt_weight,

                    "confidence":
                        confidence,

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
                }
            )

    # Select using validation mAP50-95
    best = max(
        results,
        key=lambda x: x["map50_95"]
    )

    print("\n" + "=" * 65)
    print(
        "BEST VALIDATION CONFIGURATION"
    )
    print("=" * 65)

    print(
        f"YOLOv11 weight : "
        f"{best['yolo_weight']}"
    )

    print(
        f"RT-DETR weight : "
        f"{best['rt_weight']}"
    )

    print(
        f"Confidence     : "
        f"{best['confidence']}"
    )

    print(
        f"Precision      : "
        f"{best['precision']:.4f}"
    )

    print(
        f"Recall         : "
        f"{best['recall']:.4f}"
    )

    print(
        f"F1             : "
        f"{best['f1']:.4f}"
    )

    print(
        f"mAP50          : "
        f"{best['map50']:.4f}"
    )

    print(
        f"mAP50-95       : "
        f"{best['map50_95']:.4f}"
    )

    return (
        best,
        results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    best,
    results
):

    output_file = (
        OUTPUT_DIR
        /
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
            "Dataset Split: 80:10:10\n\n"
        )

        f.write(
            "BEST VALIDATION CONFIGURATION\n"
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
            f"{best['map50_95']:.6f}\n\n"
        )

        f.write(
            "ALL VALIDATION CONFIGURATIONS\n"
        )

        f.write(
            "-" * 90 + "\n"
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
        generate_validation_predictions()
    )

    best, results = optimize_wbf(
        validation_predictions
    )

    save_results(
        best,
        results
    )

    print("\n" + "=" * 65)
    print(
        "80:10:10 WBF VALIDATION COMPLETED"
    )
    print("=" * 65)

    print(
        "\nThe test set has NOT been used."
    )

    print(
        "The selected configuration is now "
        "locked for the final test evaluation."
    )