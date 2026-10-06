# Bounding box del soggetto principale su un video qualsiasi, con YOLO26s e BoT-SORT.

from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

import paths

TRACKER = "botsort.yaml"
CONF = 0.25


# Larghezza, altezza, fps e numero di frame dichiarati dal video.
def video_meta(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"impossibile aprire il video: {video_path}")
    meta = dict(width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                fps=float(cap.get(cv2.CAP_PROP_FPS)),
                n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return meta


# Riempie i frame senza rilevamento interpolando le box valide.
def _interpolate(bboxes, mask):
    idx = np.flatnonzero(mask)
    all_idx = np.arange(len(mask))
    out = np.empty_like(bboxes)
    for c in range(4):
        out[:, c] = np.interp(all_idx, idx, bboxes[idx, c])
    return out


# Traccia le persone e restituisce box, mask e metadati della traccia più grande.
def extract_subject(video_path, device=None):
    meta = video_meta(video_path)
    tracks = defaultdict(list)

    stream = YOLO(str(paths.YOLO_SUBJECT)).track(
        source=str(video_path), classes=[0], persist=True, tracker=TRACKER,
        stream=True, verbose=False, device=device, conf=CONF)
    n = 0
    for i, r in enumerate(stream):
        n = i + 1
        if r.boxes is None or r.boxes.id is None:
            continue
        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy().astype(int)
        for box, tid in zip(boxes, ids):
            tracks[int(tid)].append((i, box.astype(np.float32)))

    meta["n_frames"] = n
    if not tracks:
        raise RuntimeError(f"nessuna persona rilevata in {video_path}")

    # Somma delle aree delle box di una traccia.
    def area_cumulata(tid):
        return sum((b[2] - b[0]) * (b[3] - b[1]) for _, b in tracks[tid])

    main_id = max(tracks, key=area_cumulata)
    bboxes = np.zeros((n, 4), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    for fi, box in tracks[main_id]:
        bboxes[fi] = box
        mask[fi] = True

    print(f"[subject] traccia {main_id}: soggetto in {int(mask.sum())}/{n} frame")
    return _interpolate(bboxes, mask), mask, meta
