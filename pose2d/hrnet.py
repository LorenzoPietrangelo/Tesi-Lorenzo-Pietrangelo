# Keypoint 2D con HRNet-w48 384x288, dalla demo di PoseFormerV2.

from types import SimpleNamespace

import cv2
import numpy as np
import torch

import paths
from shared.wrapper import Handle

from lib.hrnet.lib.config import cfg, update_config
from lib.hrnet.lib.models import pose_hrnet
from lib.hrnet.lib.utils.inference import get_final_preds
from lib.hrnet.lib.utils.transforms import get_affine_transform

PIXEL_STD = 200.0
PADDING = 1.25
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# Carica HRNet con la configurazione e il checkpoint della demo.
def load(device="cuda"):
    paths.require(paths.CFG_HRNET, "config HRNet")
    paths.require(paths.CKPT_HRNET, "checkpoint HRNet")

    update_config(cfg, SimpleNamespace(cfg=str(paths.CFG_HRNET), opts=[],
                                       modelDir=""))
    model = pose_hrnet.get_pose_net(cfg, is_train=False)
    sd = torch.load(paths.CKPT_HRNET, map_location="cpu", weights_only=True)
    sd = sd.get("state_dict", sd)
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)

    h = Handle(model.to(device).eval(), device=device)
    h.input_size = tuple(cfg.MODEL.IMAGE_SIZE)
    return h


# Ritaglia la box allargata al rapporto d'aspetto del modello e la normalizza.
def _crop(frame_bgr, box, model_w, model_h):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    center = np.array([x1 + w * 0.5, y1 + h * 0.5], dtype=np.float32)

    aspect = model_w / model_h
    if w > aspect * h:
        h = w / aspect
    else:
        w = h * aspect
    scale = np.array([w / PIXEL_STD, h / PIXEL_STD], dtype=np.float32) * PADDING

    trans = get_affine_transform(center, scale, 0, [model_w, model_h])
    patch = cv2.warpAffine(frame_bgr, trans, (model_w, model_h),
                           flags=cv2.INTER_LINEAR)
    patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)

    x = (patch.astype(np.float32) / 255.0 - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None]), center, scale


# Esegue HRNet su un ritaglio e riporta i keypoint nel frame.
@torch.no_grad()
def _infer(handle, x, center, scale):
    heatmaps = handle.model(torch.from_numpy(x).to(handle.device))
    preds, maxvals = get_final_preds(cfg, heatmaps.cpu().numpy(),
                                     np.asarray([center]), np.asarray([scale]))
    return preds[0], maxvals[0, :, 0]


# Stima i keypoint COCO-17 di ogni frame nella box del soggetto.
def run(handle, video_path, bboxes, mask):
    mw, mh = handle.input_size
    n = len(bboxes)
    kpts = np.zeros((n, 17, 2), dtype=np.float32)
    scores = np.zeros((n, 17), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            break
        x, center, scale = _crop(frame, bboxes[i], mw, mh)
        kpts[i], scores[i] = _infer(handle, x, center, scale)
    cap.release()

    return kpts[:n], scores[:n], np.asarray(mask[:n], dtype=bool)
