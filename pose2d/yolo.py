# Keypoint 2D con YOLO26x-pose, che lavora sul frame intero.

import numpy as np
from ultralytics import YOLO

import paths
from shared.wrapper import Handle

CONF = 0.25
IMGSZ = 640
MIN_IOU = 0.30


# IoU fra una box e un insieme di box, in formato xyxy.
def _iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = (np.clip(boxes[:, 2] - boxes[:, 0], 0, None) *
              np.clip(boxes[:, 3] - boxes[:, 1], 0, None))

    union = area_a + area_b - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)


# Carica YOLO26x-pose; ultralytics scarica i pesi al primo uso.
def load(device="cuda"):
    return Handle(YOLO(str(paths.YOLO_POSE)), device=device, conf=CONF,
                  imgsz=IMGSZ, min_iou=MIN_IOU)


# Per ogni frame tiene la posa con IoU massima rispetto alla box del soggetto.
def run(handle, video_path, bboxes, mask):
    n = len(bboxes)
    kpts = np.zeros((n, 17, 2), dtype=np.float32)
    scores = np.zeros((n, 17), dtype=np.float32)
    found = np.zeros(n, dtype=bool)

    stream = handle.model.predict(source=str(video_path), classes=[0],
                                  conf=CONF, imgsz=IMGSZ, device=handle.device,
                                  stream=True, verbose=False)
    letti = 0
    for i, r in enumerate(stream):
        if i >= n:
            break
        letti = i + 1
        if r.keypoints is None or len(r.boxes) == 0:
            continue

        iou = _iou(bboxes[i], r.boxes.xyxy.cpu().numpy())
        j = int(np.argmax(iou))
        if iou[j] < MIN_IOU:
            continue

        kpts[i] = r.keypoints.xy[j].cpu().numpy()
        if r.keypoints.conf is not None:
            scores[i] = r.keypoints.conf[j].cpu().numpy()
        found[i] = True

    n = letti
    return kpts[:n], scores[:n], np.asarray(mask[:n], dtype=bool) & found[:n]
