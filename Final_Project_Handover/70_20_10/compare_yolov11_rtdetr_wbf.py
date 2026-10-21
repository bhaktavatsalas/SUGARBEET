from pathlib import Path
import yaml
import numpy as np
from ultralytics import YOLO
from ensemble_boxes import weighted_boxes_fusion


# ============================================================
# FAIR COMPARISON
# YOLOv11 vs RT-DETR vs YOLOv11 + RT-DETR WBF
#
# Dataset: 70:20:10
# Evaluation: TEST SET
#
# WBF configuration locked from validation:
# YOLOv11 = 0.2
# RT-DETR = 0.8
# Confidence threshold = 0.25
# ============================================================


# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

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

OUTPUT_DIR = (
    BASE / "YOLOv11_RTDETR_WBF_70_20_10"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ------------------------------------------------------------
# LOCKED WBF SETTINGS
# ------------------------------------------------------------

YOLO_WEIGHT = 0.2
RT_WEIGHT = 0.8

CONF_THRESHOLD = 0.25

WBF_IOU_THRESHOLD = 0.55


# ------------------------------------------------------------
# LOAD YAML
# ------------------------------------------------------------

with open(DATA_YAML, "r") as f:
    data = yaml.safe_load(f)

DATASET_PATH = Path(data["path"])

TEST_IMAGES = (
    DATASET_PATH / data["test"]
)

TEST_LABELS = (
    DATASET_PATH / "labels/test"
)


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
# GET MODEL PREDICTIONS
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
# READ GROUND TRUTH
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
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
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
    confidence_threshold=0.25,
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
        f1,
        total_tp,
        total_fp,
        total_fn
    )


# ============================================================
# AP
# ============================================================

def calculate_ap(
    predictions,
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

            if score >= 0.001:

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

    for i, pred in enumerate(
        all_predictions
    ):

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

            ap_values.append(
                0.0
            )

    return float(
        np.mean(ap_values)
    )


# ============================================================
# CALCULATE mAP50 AND mAP50-95
# ============================================================

def calculate_map(
    predictions
):

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

    map50_95 = np.mean(
        aps
    )

    return (
        map50,
        map50_95
    )


# ============================================================
# GENERATE TEST PREDICTIONS
# ============================================================

def generate_predictions():

    print("\n" + "=" * 65)
    print(
        "GENERATING TEST PREDICTIONS"
    )
    print("=" * 65)

    image_files = sorted(
        list(
            TEST_IMAGES.glob("*.jpg")
        )
        +
        list(
            TEST_IMAGES.glob("*.jpeg")
        )
        +
        list(
            TEST_IMAGES.glob("*.png")
        )
    )

    print(
        f"Test images: "
        f"{len(image_files)}"
    )

    yolo_predictions = []
    rt_predictions = []
    wbf_predictions = []

    for index, image_path in enumerate(
        image_files,
        start=1
    ):

        print(
            f"\rProcessing "
            f"{index}/{len(image_files)}",
            end=""
        )

        label_path = (
            TEST_LABELS
            /
            f"{image_path.stem}.txt"
        )

        gt_boxes = read_ground_truth(
            label_path
        )

        # ----------------------------------------
        # YOLOv11
        # ----------------------------------------

        (
            yolo_boxes,
            yolo_scores,
            yolo_labels
        ) = get_predictions(
            yolo_model,
            image_path
        )

        # ----------------------------------------
        # RT-DETR
        # ----------------------------------------

        (
            rt_boxes,
            rt_scores,
            rt_labels
        ) = get_predictions(
            rtdetr_model,
            image_path
        )

        # ----------------------------------------
        # WBF
        # ----------------------------------------

        (
            fused_boxes,
            fused_scores,
            fused_labels
        ) = fuse_predictions(
            yolo_boxes,
            yolo_scores,
            yolo_labels,
            rt_boxes,
            rt_scores,
            rt_labels
        )

        yolo_predictions.append(
            {
                "image_id":
                    image_path.stem,

                "boxes":
                    yolo_boxes,

                "scores":
                    yolo_scores,

                "labels":
                    yolo_labels,

                "gt_boxes":
                    gt_boxes
            }
        )

        rt_predictions.append(
            {
                "image_id":
                    image_path.stem,

                "boxes":
                    rt_boxes,

                "scores":
                    rt_scores,

                "labels":
                    rt_labels,

                "gt_boxes":
                    gt_boxes
            }
        )

        wbf_predictions.append(
            {
                "image_id":
                    image_path.stem,

                "boxes":
                    fused_boxes,

                "scores":
                    fused_scores,

                "labels":
                    fused_labels,

                "gt_boxes":
                    gt_boxes
            }
        )

    print(
        "\n\nAll test predictions generated."
    )

    return (
        yolo_predictions,
        rt_predictions,
        wbf_predictions
    )


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    name,
    predictions
):

    print("\n" + "-" * 65)
    print(name)
    print("-" * 65)

    (
        precision,
        recall,
        f1,
        tp,
        fp,
        fn
    ) = evaluate_prf(
        predictions,
        CONF_THRESHOLD,
        0.50
    )

    (
        map50,
        map50_95
    ) = calculate_map(
        predictions
    )

    print(
        f"Precision : {precision:.4f}"
    )

    print(
        f"Recall    : {recall:.4f}"
    )

    print(
        f"F1        : {f1:.4f}"
    )

    print(
        f"mAP50     : {map50:.4f}"
    )

    print(
        f"mAP50-95  : {map50_95:.4f}"
    )

    print(
        f"TP        : {tp}"
    )

    print(
        f"FP        : {fp}"
    )

    print(
        f"FN        : {fn}"
    )

    return {
        "model": name,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "map50": map50,
        "map50_95": map50_95,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print(
        "FAIR TEST COMPARISON"
    )
    print(
        "YOLOv11 vs RT-DETR vs WBF"
    )
    print(
        "Dataset: 70:20:10"
    )
    print("=" * 65)

    print(
        f"\nEvaluation confidence: "
        f"{CONF_THRESHOLD}"
    )

    print(
        f"WBF weights: "
        f"YOLOv11={YOLO_WEIGHT}, "
        f"RT-DETR={RT_WEIGHT}"
    )

    (
        yolo_predictions,
        rt_predictions,
        wbf_predictions
    ) = generate_predictions()

    # --------------------------------------------------------
    # Evaluate all three
    # --------------------------------------------------------

    yolo_result = evaluate_model(
        "YOLOv11",
        yolo_predictions
    )

    rt_result = evaluate_model(
        "RT-DETR",
        rt_predictions
    )

    wbf_result = evaluate_model(
        "YOLOv11 + RT-DETR WBF",
        wbf_predictions
    )

    # --------------------------------------------------------
    # Final table
    # --------------------------------------------------------

    results = [
        yolo_result,
        rt_result,
        wbf_result
    ]

    print("\n" + "=" * 65)
    print(
        "FINAL FAIR COMPARISON"
    )
    print("=" * 65)

    print(
        "\nModel                      "
        "P       R       F1      "
        "mAP50   mAP50-95"
    )

    print("-" * 65)

    for r in results:

        print(
            f"{r['model']:<26}"
            f"{r['precision']:.4f}  "
            f"{r['recall']:.4f}  "
            f"{r['f1']:.4f}  "
            f"{r['map50']:.4f}  "
            f"{r['map50_95']:.4f}"
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        /
        "fair_70_20_10_test_comparison.txt"
    )

    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            "FAIR TEST COMPARISON\n"
        )

        f.write(
            "Dataset: 70:20:10\n"
        )

        f.write(
            "Evaluation Set: TEST\n\n"
        )

        f.write(
            f"Confidence Threshold: "
            f"{CONF_THRESHOLD}\n"
        )

        f.write(
            f"WBF YOLOv11 Weight: "
            f"{YOLO_WEIGHT}\n"
        )

        f.write(
            f"WBF RT-DETR Weight: "
            f"{RT_WEIGHT}\n\n"
        )

        for r in results:

            f.write(
                f"{r['model']}\n"
            )

            f.write(
                f"Precision: "
                f"{r['precision']:.6f}\n"
            )

            f.write(
                f"Recall: "
                f"{r['recall']:.6f}\n"
            )

            f.write(
                f"F1: "
                f"{r['f1']:.6f}\n"
            )

            f.write(
                f"mAP50: "
                f"{r['map50']:.6f}\n"
            )

            f.write(
                f"mAP50-95: "
                f"{r['map50_95']:.6f}\n"
            )

            f.write(
                f"TP: {r['tp']}\n"
            )

            f.write(
                f"FP: {r['fp']}\n"
            )

            f.write(
                f"FN: {r['fn']}\n\n"
            )

    print("\n" + "=" * 65)

    print(
        "COMPARISON COMPLETED"
    )

    print("=" * 65)

    print(
        "\nSaved to:"
    )

    print(
        output_file
    )