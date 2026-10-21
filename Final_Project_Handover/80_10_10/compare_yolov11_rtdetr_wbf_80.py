from pathlib import Path
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# PATHS
# ============================================================

BASE = Path("/Users/kalamanjunatha/Downloads/sugarbeet/Experiments/80_10_10")

DATASET = Path(
    "/Users/kalamanjunatha/Downloads/sugarbeet/dataset_80_10_10_detection"
)

YOLOV11_MODEL = BASE / "YOLOv11/detection/weights/best.pt"
RTDETR_MODEL = BASE / "RT_DETR4/weights/best.pt"

TEST_IMAGES = DATASET / "images/test"
TEST_LABELS = DATASET / "labels/test"

# Locked WBF configuration obtained from validation
WBF_YOLOV11_WEIGHT = 0.3
WBF_RTDETR_WEIGHT = 0.7
CONF_THRESHOLD = 0.25
WBF_IOU = 0.55


# ============================================================
# FUNCTIONS
# ============================================================

def load_ground_truth(label_path, image_width, image_height):
    """
    Load YOLO-format ground-truth boxes and convert them to
    xyxy pixel coordinates.
    """

    boxes = []

    if not label_path.exists():
        return boxes

    with open(label_path, "r") as f:
        for line in f:
            values = line.strip().split()

            if len(values) < 5:
                continue

            cls = int(values[0])
            x_center = float(values[1])
            y_center = float(values[2])
            width = float(values[3])
            height = float(values[4])

            x1 = (x_center - width / 2) * image_width
            y1 = (y_center - height / 2) * image_height
            x2 = (x_center + width / 2) * image_width
            y2 = (y_center + height / 2) * image_height

            boxes.append([x1, y1, x2, y2, cls])

    return boxes


def calculate_iou(box1, box2):
    """
    Calculate IoU between two xyxy boxes.
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def evaluate_predictions(all_predictions, all_ground_truths, iou_threshold=0.5):
    """
    Calculate TP, FP, FN, Precision, Recall and F1.
    """

    TP = 0
    FP = 0
    FN = 0

    for predictions, ground_truths in zip(
        all_predictions, all_ground_truths
    ):

        matched_gt = set()

        # Highest confidence predictions first
        predictions = sorted(
            predictions,
            key=lambda x: x[4],
            reverse=True
        )

        for pred in predictions:

            best_iou = 0.0
            best_gt = -1

            for gt_index, gt in enumerate(ground_truths):

                if gt_index in matched_gt:
                    continue

                # Same class
                if int(pred[5]) != int(gt[4]):
                    continue

                iou = calculate_iou(pred, gt)

                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_index

            if best_iou >= iou_threshold and best_gt >= 0:
                TP += 1
                matched_gt.add(best_gt)
            else:
                FP += 1

        FN += len(ground_truths) - len(matched_gt)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    return precision, recall, f1, TP, FP, FN


def calculate_ap(all_predictions, all_ground_truths, iou_threshold):
    """
    Calculate AP using confidence-ranked predictions.
    """

    prediction_list = []

    total_gt = 0

    for image_index, (predictions, ground_truths) in enumerate(
        zip(all_predictions, all_ground_truths)
    ):

        total_gt += len(ground_truths)

        for pred in predictions:
            prediction_list.append(
                (
                    pred[4],
                    image_index,
                    pred
                )
            )

    if total_gt == 0:
        return 0.0

    prediction_list.sort(key=lambda x: x[0], reverse=True)

    matched = {
        i: set()
        for i in range(len(all_ground_truths))
    }

    tp = np.zeros(len(prediction_list))
    fp = np.zeros(len(prediction_list))

    for index, (_, image_index, pred) in enumerate(prediction_list):

        ground_truths = all_ground_truths[image_index]

        best_iou = 0.0
        best_gt = -1

        for gt_index, gt in enumerate(ground_truths):

            if gt_index in matched[image_index]:
                continue

            if int(pred[5]) != int(gt[4]):
                continue

            iou = calculate_iou(pred, gt)

            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index

        if best_iou >= iou_threshold and best_gt >= 0:
            tp[index] = 1
            matched[image_index].add(best_gt)
        else:
            fp[index] = 1

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recalls = cumulative_tp / total_gt
    precisions = cumulative_tp / (
        cumulative_tp + cumulative_fp + 1e-9
    )

    # Precision envelope
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(
            precisions[i],
            precisions[i + 1]
        )

    recall_points = np.concatenate(
        ([0.0], recalls, [1.0])
    )

    precision_points = np.concatenate(
        ([1.0], precisions, [0.0])
    )

    ap = 0.0

    for i in range(len(recall_points) - 1):
        ap += (
            recall_points[i + 1] - recall_points[i]
        ) * precision_points[i + 1]

    return float(ap)


def calculate_map(all_predictions, all_ground_truths):
    """
    Calculate mAP@50 and mAP@50-95.
    """

    ap50 = calculate_ap(
        all_predictions,
        all_ground_truths,
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
                all_predictions,
                all_ground_truths,
                threshold
            )
        )

    map5095 = float(np.mean(aps))

    return ap50, map5095


def get_model_predictions(model, image_path):
    """
    Generate predictions in the common format:
    [x1, y1, x2, y2, confidence, class]
    """

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
        predictions.append(
            [
                float(box[0]),
                float(box[1]),
                float(box[2]),
                float(box[3]),
                float(score),
                int(cls)
            ]
        )

    return predictions


def create_wbf_predictions(
    yolo_predictions,
    rtdetr_predictions,
    image_width,
    image_height
):
    """
    Perform Weighted Box Fusion.
    """

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

            boxes.append(
                [
                    max(0, min(1, x1 / image_width)),
                    max(0, min(1, y1 / image_height)),
                    max(0, min(1, x2 / image_width)),
                    max(0, min(1, y2 / image_height))
                ]
            )

            scores.append(score)
            labels.append(int(cls))

        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    if not any(boxes_list):
        return []

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=[
            WBF_YOLOV11_WEIGHT,
            WBF_RTDETR_WEIGHT
        ],
        iou_thr=WBF_IOU,
        skip_box_thr=CONF_THRESHOLD
    )

    final_predictions = []

    for box, score, cls in zip(
        fused_boxes,
        fused_scores,
        fused_labels
    ):

        x1 = box[0] * image_width
        y1 = box[1] * image_height
        x2 = box[2] * image_width
        y2 = box[3] * image_height

        final_predictions.append(
            [
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                float(score),
                int(cls)
            ]
        )

    return final_predictions


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("YOLOv11 + RT-DETR WBF")
print("FAIR 80:10:10 TEST COMPARISON")
print("=" * 70)

print()
print("Dataset:")
print(DATASET)

print()
print("YOLOv11:")
print(YOLOV11_MODEL)

print()
print("RT-DETR:")
print(RTDETR_MODEL)

print()
print("Evaluation configuration:")
print("Confidence threshold :", CONF_THRESHOLD)
print("WBF IoU threshold    :", WBF_IOU)
print("YOLOv11 WBF weight   :", WBF_YOLOV11_WEIGHT)
print("RT-DETR WBF weight   :", WBF_RTDETR_WEIGHT)

# Verify paths
if not YOLOV11_MODEL.exists():
    raise FileNotFoundError(
        f"YOLOv11 model not found: {YOLOV11_MODEL}"
    )

if not RTDETR_MODEL.exists():
    raise FileNotFoundError(
        f"RT-DETR model not found: {RTDETR_MODEL}"
    )

if not TEST_IMAGES.exists():
    raise FileNotFoundError(
        f"Test image folder not found: {TEST_IMAGES}"
    )

if not TEST_LABELS.exists():
    raise FileNotFoundError(
        f"Test label folder not found: {TEST_LABELS}"
    )

print()
print("All paths verified.")

# Load models
print()
print("Loading YOLOv11...")
yolo11 = YOLO(str(YOLOV11_MODEL))

print("Loading RT-DETR...")
rtdetr = YOLO(str(RTDETR_MODEL))

print("Models loaded successfully.")

# Get test images
image_files = sorted(
    list(TEST_IMAGES.glob("*.jpg")) +
    list(TEST_IMAGES.glob("*.jpeg")) +
    list(TEST_IMAGES.glob("*.png"))
)

print()
print(f"Test images: {len(image_files)}")

if len(image_files) == 0:
    raise RuntimeError("No test images found.")

# Storage
yolo11_predictions = []
rtdetr_predictions = []
wbf_predictions = []
ground_truths = []

# ============================================================
# GENERATE PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("GENERATING TEST PREDICTIONS")
print("=" * 70)

for index, image_path in enumerate(image_files, start=1):

    print(
        f"Processing test image "
        f"{index}/{len(image_files)}",
        end="\r"
    )

    # YOLOv11
    pred_yolo = get_model_predictions(
        yolo11,
        image_path
    )

    # RT-DETR
    pred_rtdetr = get_model_predictions(
        rtdetr,
        image_path
    )

    # Read image dimensions
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size

    # Ground truth
    label_path = TEST_LABELS / f"{image_path.stem}.txt"

    gt = load_ground_truth(
        label_path,
        width,
        height
    )

    # WBF
    pred_wbf = create_wbf_predictions(
        pred_yolo,
        pred_rtdetr,
        width,
        height
    )

    yolo11_predictions.append(pred_yolo)
    rtdetr_predictions.append(pred_rtdetr)
    wbf_predictions.append(pred_wbf)
    ground_truths.append(gt)

print()
print()
print("Test predictions generated.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(name, predictions):

    precision, recall, f1, tp, fp, fn = evaluate_predictions(
        predictions,
        ground_truths,
        iou_threshold=0.5
    )

    map50, map5095 = calculate_map(
        predictions,
        ground_truths
    )

    print()
    print("-" * 70)
    print(name)
    print("-" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"mAP50     : {map50:.4f}")
    print(f"mAP50-95  : {map5095:.4f}")

    print()
    print(f"TP        : {tp}")
    print(f"FP        : {fp}")
    print(f"FN        : {fn}")

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map5095": map5095,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# FAIR TEST COMPARISON
# ============================================================

results = []

results.append(
    evaluate_model(
        "YOLOv11",
        yolo11_predictions
    )
)

results.append(
    evaluate_model(
        "RT-DETR",
        rtdetr_predictions
    )
)

results.append(
    evaluate_model(
        "YOLOv11 + RT-DETR WBF",
        wbf_predictions
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL 80:10:10 FAIR TEST COMPARISON")
print("=" * 70)

print()
print(
    f"{'Model':<25}"
    f"{'P':>9}"
    f"{'R':>9}"
    f"{'F1':>9}"
    f"{'mAP50':>10}"
    f"{'mAP50-95':>12}"
)

print("-" * 70)

for result in results:

    print(
        f"{result['model']:<25}"
        f"{result['precision']:>9.4f}"
        f"{result['recall']:>9.4f}"
        f"{result['f1']:>9.4f}"
        f"{result['map50']:>10.4f}"
        f"{result['map5095']:>12.4f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

OUTPUT_DIR = BASE / "YOLOv11_RTDETR_WBF_80_10_10"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = OUTPUT_DIR / "fair_test_comparison_results.txt"

with open(OUTPUT_FILE, "w") as f:

    f.write(
        "80:10:10 FAIR TEST COMPARISON\n"
    )

    f.write(
        "YOLOv11 vs RT-DETR vs YOLOv11 + RT-DETR WBF\n\n"
    )

    f.write(
        f"Confidence threshold: {CONF_THRESHOLD}\n"
    )

    f.write(
        f"WBF IoU threshold: {WBF_IOU}\n"
    )

    f.write(
        f"YOLOv11 WBF weight: {WBF_YOLOV11_WEIGHT}\n"
    )

    f.write(
        f"RT-DETR WBF weight: {WBF_RTDETR_WEIGHT}\n\n"
    )

    for result in results:

        f.write(
            f"{result['model']}\n"
        )

        f.write(
            f"Precision: {result['precision']:.4f}\n"
        )

        f.write(
            f"Recall: {result['recall']:.4f}\n"
        )

        f.write(
            f"F1: {result['f1']:.4f}\n"
        )

        f.write(
            f"mAP50: {result['map50']:.4f}\n"
        )

        f.write(
            f"mAP50-95: {result['map5095']:.4f}\n"
        )

        f.write(
            f"TP: {result['tp']}\n"
        )

        f.write(
            f"FP: {result['fp']}\n"
        )

        f.write(
            f"FN: {result['fn']}\n\n"
        )

print()
print("=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)

print()
print("Results saved to:")
print(OUTPUT_FILE)