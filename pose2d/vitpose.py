# Keypoint 2D con ViTPose+ huge tramite transformers.

import cv2
import numpy as np
import torch
from transformers import AutoProcessor, VitPoseForPoseEstimation

from shared.wrapper import Handle

MODEL_ID = "usyd-community/vitpose-plus-huge"
DATASET_INDEX = 0


# Carica ViTPose+ e il suo processore, in fp16 se si usa la GPU.
def load(device="cuda"):
    half = str(device).startswith("cuda")
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = VitPoseForPoseEstimation.from_pretrained(
        MODEL_ID, dtype=torch.float16 if half else torch.float32)

    h = Handle(model.to(device).eval(), device=device, model_id=MODEL_ID,
               half=half)
    h.processor = processor
    return h


# Stima la posa nella box di un frame, con l'esperto COCO del modello.
@torch.no_grad()
def _infer(handle, frame_rgb, box_xywh):
    inputs = handle.processor(frame_rgb, boxes=[[box_xywh]],
                              return_tensors="pt").to(handle.device)
    inputs["pixel_values"] = inputs["pixel_values"].to(handle.model.dtype)

    outputs = handle.model(
        **inputs,
        dataset_index=torch.tensor([DATASET_INDEX], device=handle.device))
    outputs.heatmaps = outputs.heatmaps.float()

    result = handle.processor.post_process_pose_estimation(outputs,
                                                           boxes=[[box_xywh]])
    return result[0][0]


# Stima i keypoint COCO-17 di ogni frame nella box del soggetto.
def run(handle, video_path, bboxes, mask):
    n = len(bboxes)
    kpts = np.zeros((n, 17, 2), dtype=np.float32)
    scores = np.zeros((n, 17), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            break
        x1, y1, x2, y2 = bboxes[i]
        box_xywh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

        pose = _infer(handle, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box_xywh)
        kpts[i] = pose["keypoints"].cpu().numpy()
        scores[i] = pose["scores"].cpu().numpy()
    cap.release()

    return kpts[:n], scores[:n], np.asarray(mask[:n], dtype=bool)
