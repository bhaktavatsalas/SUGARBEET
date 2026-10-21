from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion
from PIL import Image


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/70_20_10"
)

DATASET = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/dataset_70_20_10_detection"
)

YOLOV12_MODEL = BASE / \
    "YOLOv12_70_20_10_Detection/train/weights/best.pt"

RTDETR_MODEL = BASE / \
    "RT_DETR_70_20_10/weights/best.pt"

VAL_IMAGES = DATASET / "images/val"
VAL_LABELS = DATASET / "labels/val"

OUTPUT_DIR = BASE / "YOLOv12_RTDETR_WBF_70_20_10"


# ============================================================
# WBF SETTINGS
# ============================================================

CONF_THRESHOLD = 0.25
WBF_IOU = 0.55

WEIGHT_CANDIDATES = [
    (0.2, 0.8),
    (0.3, 0.7),
    (0.4, 0.6),
    (0.5, 0.5),
    (0.6, 0.4),
    (0.7, 0.3),
    (0.8, 0.2),
]

CONF_CANDIDATES = [
    0.25,
    0.35,
    0.45,
    0.55,
]


# ============================================================
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)

    intersection = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * \
            max(0, box1[3] - box1[1])

    area2 = max(0, box2[2] - box2[0]) * \
            max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GROUND TRUTH
# ============================================================

def load_ground_truth(label_path, width, height):

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:

        for line in f:

            values = line.strip().split()

            if len(values) < 5:
                continue

            cls = int(values[0])

            xc = float(values[1])
            yc = float(values[2])
            w = float(values[3])
            h = float(values[4])

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            boxes.append([
                x1, y1, x2, y2, cls
            ])

    return boxes


# ============================================================
# MODEL PREDICTIONS
# ============================================================

def get_predictions(model, image_path):

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONF_THRESHOLD,
        verbose=False
    )[0]

    predictions = []

    if result.boxes is None:
        return predictions

    boxes = result.boxes.xyxy.cpu().numpy()
    scores = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    for box, score, cls in zip(
        boxes,
        scores,
        classes
    ):

        predictions.append([
            float(box[0]),
            float(box[1]),
            float(box[2]),
            float(box[3]),
            float(score),
            int(cls)
        ])

    return predictions


# ============================================================
# WBF
# ============================================================

def create_wbf(
    yolo_predictions,
    rtdetr_predictions,
    width,
    height,
    yolo_weight,
    rtdetr_weight
):

    boxes_list = []
    scores_list = []
    labels_list = []

    for predictions in [
        yolo_predictions,
        rtdetr_predictions
    ]:

        boxes = []
        scores = []
        labels = []

        for pred in predictions:

            x1, y1, x2, y2, score, cls = pred

            boxes.append([
                max(0, min(1, x1 / width)),
                max(0, min(1, y1 / height)),
                max(0, min(1, x2 / width)),
                max(0, min(1, y2 / height))
            ])

            scores.append(score)
            labels.append(int(cls))

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if not any(boxes_list):
        return []

    fused_boxes, fused_scores, fused_labels = \
        weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=[
                yolo_weight,
                rtdetr_weight
            ],
            iou_thr=WBF_IOU,
            skip_box_thr=CONF_THRESHOLD
        )

    predictions = []

    for box, score, cls in zip(
        fused_boxes,
        fused_scores,
        fused_labels
    ):

        predictions.append([
            float(box[0] * width),
            float(box[1] * height),
            float(box[2] * width),
            float(box[3] * height),
            float(score),
            int(cls)
        ])

    return predictions


# ============================================================
# EVALUATION
# ============================================================

def evaluate_predictions(
    predictions,
    ground_truths,
    iou_threshold=0.5
):

    TP = 0
    FP = 0
    FN = 0

    for preds, gts in zip(
        predictions,
        ground_truths
    ):

        matched = set()

        preds = sorted(
            preds,
            key=lambda x: x[4],
            reverse=True
        )

        for pred in preds:

            best_iou = 0
            best_gt = -1

            for i, gt in enumerate(gts):

                if i in matched:
                    continue

                if int(pred[5]) != int(gt[4]):
                    continue

                iou = calculate_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_gt = i

            if best_iou >= iou_threshold:

                TP += 1
                matched.add(best_gt)

            else:

                FP += 1

        FN += len(gts) - len(matched)

    precision = (
        TP / (TP + FP)
        if TP + FP > 0 else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN > 0 else 0
    )

    f1 = (
        2 * precision * recall /
        (precision + recall)
        if precision + recall > 0 else 0
    )

    return precision, recall, f1


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

    for image_index, (preds, gts) in enumerate(
        zip(predictions, ground_truths)
    ):

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
        return 0

    prediction_list.sort(
        key=lambda x: x[0],
        reverse=True
    )

    matched = {
        i: set()
        for i in range(len(ground_truths))
    }

    tp = np.zeros(len(prediction_list))
    fp = np.zeros(len(prediction_list))

    for index, (_, image_index, pred) in enumerate(
        prediction_list
    ):

        gts = ground_truths[image_index]

        best_iou = 0
        best_gt = -1

        for gt_index, gt in enumerate(gts):

            if gt_index in matched[image_index]:
                continue

            if int(pred[5]) != int(gt[4]):
                continue

            iou = calculate_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index

        if best_iou >= iou_threshold:

            tp[index] = 1
            matched[image_index].add(best_gt)

        else:

            fp[index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / total_gt

    precisions = cumulative_tp / (
        cumulative_tp +
        cumulative_fp +
        1e-9
    )

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
        ([0], recalls, [1])
    )

    precision_points = np.concatenate(
        ([1], precisions, [0])
    )

    ap = 0

    for i in range(
        len(recall_points) - 1
    ):

        ap += (
            recall_points[i + 1] -
            recall_points[i]
        ) * precision_points[i + 1]

    return float(ap)


def calculate_map(
    predictions,
    ground_truths
):

    map50 = calculate_ap(
        predictions,
        ground_truths,
        0.50
    )

    thresholds = np.arange(
        0.50,
        0.96,
        0.05
    )

    aps = []

    for threshold in thresholds:

        aps.append(
            calculate_ap(
                predictions,
                ground_truths,
                threshold
            )
        )

    map5095 = float(np.mean(aps))

    return map50, map5095


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("YOLOv12 + RT-DETR WBF")
print("70:20:10 VALIDATION EVALUATION")
print("=" * 70)

print()
print("YOLOv12:")
print(YOLOV12_MODEL)

print()
print("RT-DETR:")
print(RTDETR_MODEL)

if not YOLOV12_MODEL.exists():
    raise FileNotFoundError(
        f"YOLOv12 model not found:\n{YOLOV12_MODEL}"
    )

if not RTDETR_MODEL.exists():
    raise FileNotFoundError(
        f"RT-DETR model not found:\n{RTDETR_MODEL}"
    )

print()
print("All model paths verified.")

print()
print("Loading YOLOv12...")
yolo12 = YOLO(str(YOLOV12_MODEL))

print("Loading RT-DETR...")
rtdetr = YOLO(str(RTDETR_MODEL))

print("Models loaded successfully.")

image_files = sorted(
    list(VAL_IMAGES.glob("*.jpg")) +
    list(VAL_IMAGES.glob("*.jpeg")) +
    list(VAL_IMAGES.glob("*.png"))
)

print()
print(f"Validation images: {len(image_files)}")


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

yolo_predictions = []
rtdetr_predictions = []
ground_truths = []

print()
print("=" * 70)
print("GENERATING VALIDATION PREDICTIONS")
print("=" * 70)

for index, image_path in enumerate(
    image_files,
    start=1
):

    print(
        f"Processing validation image "
        f"{index}/{len(image_files)}",
        end="\r"
    )

    yolo_pred = get_predictions(
        yolo12,
        image_path
    )

    rtdetr_pred = get_predictions(
        rtdetr,
        image_path
    )

    with Image.open(image_path) as img:
        width, height = img.size

    label_path = VAL_LABELS / (
        image_path.stem + ".txt"
    )

    gt = load_ground_truth(
        label_path,
        width,
        height
    )

    yolo_predictions.append(yolo_pred)
    rtdetr_predictions.append(rtdetr_pred)
    ground_truths.append(gt)

print()
print()
print("Validation predictions generated.")


# ============================================================
# SEARCH BEST WBF CONFIGURATION
# ============================================================

best_result = None
all_results = []

print()
print("=" * 70)
print("SEARCHING WBF CONFIGURATION")
print("=" * 70)

for yolo_weight, rtdetr_weight in WEIGHT_CANDIDATES:

    for confidence in CONF_CANDIDATES:

        CONF_THRESHOLD = confidence

        wbf_predictions = []

        for yolo_pred, rtdetr_pred, image_path in zip(
            yolo_predictions,
            rtdetr_predictions,
            image_files
        ):

            with Image.open(image_path) as img:
                width, height = img.size

            fused = create_wbf(
                yolo_pred,
                rtdetr_pred,
                width,
                height,
                yolo_weight,
                rtdetr_weight
            )

            fused = [
                p for p in fused
                if p[4] >= confidence
            ]

            wbf_predictions.append(fused)

        precision, recall, f1 = evaluate_predictions(
            wbf_predictions,
            ground_truths
        )

        map50, map5095 = calculate_map(
            wbf_predictions,
            ground_truths
        )

        result = {
            "yolo_weight": yolo_weight,
            "rtdetr_weight": rtdetr_weight,
            "confidence": confidence,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": map50,
            "map5095": map5095
        }

        all_results.append(result)

        print(
            f"YOLOv12={yolo_weight:.1f} "
            f"RT-DETR={rtdetr_weight:.1f} "
            f"Conf={confidence:.2f} | "
            f"P={precision:.4f} "
            f"R={recall:.4f} "
            f"F1={f1:.4f} "
            f"mAP50={map50:.4f} "
            f"mAP50-95={map5095:.4f}"
        )

        if (
            best_result is None
            or map5095 > best_result["map5095"]
        ):

            best_result = result


# ============================================================
# FINAL BEST CONFIGURATION
# ============================================================

print()
print("=" * 70)
print("BEST WBF CONFIGURATION")
print("=" * 70)

print(
    f"YOLOv12 weight : "
    f"{best_result['yolo_weight']:.1f}"
)

print(
    f"RT-DETR weight : "
    f"{best_result['rtdetr_weight']:.1f}"
)

print(
    f"Confidence     : "
    f"{best_result['confidence']:.2f}"
)

print(
    f"WBF IoU        : "
    f"{WBF_IOU:.2f}"
)

print()
print(
    f"Precision      : "
    f"{best_result['precision']:.4f}"
)

print(
    f"Recall         : "
    f"{best_result['recall']:.4f}"
)

print(
    f"F1 Score       : "
    f"{best_result['f1']:.4f}"
)

print(
    f"mAP50          : "
    f"{best_result['map50']:.4f}"
)

print(
    f"mAP50-95       : "
    f"{best_result['map5095']:.4f}"
)


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "wbf_validation_results.txt"

with open(OUTPUT_FILE, "w") as f:

    f.write(
        "YOLOv12 + RT-DETR WBF\n"
        "70:20:10 VALIDATION RESULTS\n\n"
    )

    f.write(
        f"Best YOLOv12 weight: "
        f"{best_result['yolo_weight']:.1f}\n"
    )

    f.write(
        f"Best RT-DETR weight: "
        f"{best_result['rtdetr_weight']:.1f}\n"
    )

    f.write(
        f"Best confidence: "
        f"{best_result['confidence']:.2f}\n"
    )

    f.write(
        f"WBF IoU: {WBF_IOU:.2f}\n\n"
    )

    f.write(
        f"Precision: "
        f"{best_result['precision']:.4f}\n"
    )

    f.write(
        f"Recall: "
        f"{best_result['recall']:.4f}\n"
    )

    f.write(
        f"F1: "
        f"{best_result['f1']:.4f}\n"
    )

    f.write(
        f"mAP50: "
        f"{best_result['map50']:.4f}\n"
    )

    f.write(
        f"mAP50-95: "
        f"{best_result['map5095']:.4f}\n"
    )

print()
print("=" * 70)
print("WBF VALIDATION COMPLETED")
print("=" * 70)

print()
print("Results saved to:")
print(OUTPUT_FILE)