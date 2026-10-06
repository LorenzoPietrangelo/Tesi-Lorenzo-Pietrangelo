# Posa 3D con MediaPipe BlazePose heavy, su un ritaglio quadrato attorno alla box del soggetto.

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

import paths
from shared.skeleton import coco_to_h36m, coco_to_h36m_3d, to_coco
from shared.wrapper import Handle

PADDING = 1.25
MIN_CONF = 0.5


# Handle che chiude il grafo nativo di MediaPipe.
class _Landmarker(Handle):
    # Chiude il landmarker.
    def close(self):
        self.model.close()


# Crea il Pose Landmarker in modalità video, sempre su CPU.
def load(device=None):
    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(
            model_asset_path=str(paths.require_blazepose()),
            delegate=mp_python.BaseOptions.Delegate.CPU),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=MIN_CONF,
        min_pose_presence_confidence=MIN_CONF,
        min_tracking_confidence=MIN_CONF,
        output_segmentation_masks=False,
    )
    h = _Landmarker(mp_vision.PoseLandmarker.create_from_options(options),
                    device="cpu", units="m", padding=PADDING, min_conf=MIN_CONF)
    h.ultimo_ts = -1
    return h


# Ritaglio quadrato RGB centrato sulla box, con bordo nero fuori dal frame.
def _crop(frame_bgr, box):
    h, w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = map(float, box)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(int(round(max(x2 - x1, y2 - y1) * PADDING)), 1)
    ox, oy = int(round(cx - s / 2)), int(round(cy - s / 2))

    canvas = np.zeros((s, s, 3), dtype=np.uint8)
    x0, y0 = max(ox, 0), max(oy, 0)
    x3, y3 = min(ox + s, w), min(oy + s, h)
    if x3 > x0 and y3 > y0:
        canvas[y0 - oy:y3 - oy, x0 - ox:x3 - ox] = frame_bgr[y0:y3, x0:x3]
    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), (float(ox), float(oy)), s


# Landmark nell'immagine, landmark in metri e visibilità; None se non c'è una posa.
def _landmarks(result):
    if not result.pose_landmarks or not result.pose_world_landmarks:
        return None
    img = result.pose_landmarks[0]
    wld = result.pose_world_landmarks[0]
    lm_img = np.array([[p.x, p.y, p.z] for p in img], dtype=np.float32)
    lm_wld = np.array([[p.x, p.y, p.z] for p in wld], dtype=np.float32)
    vis = np.array([1.0 if p.visibility is None else p.visibility for p in img],
                   dtype=np.float32)
    return lm_img, lm_wld, vis


# Stima la posa di ogni frame; i timestamp di MediaPipe proseguono fra un video e l'altro.
def run(handle, video_path, bboxes, mask):
    n = len(bboxes)
    lm_img = np.zeros((n, 33, 3), dtype=np.float32)
    lm_wld = np.zeros((n, 33, 3), dtype=np.float32)
    vis = np.zeros((n, 33), dtype=np.float32)
    found = np.zeros(n, dtype=bool)

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    prev_ts = handle.ultimo_ts
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            break
        patch, (ox, oy), s = _crop(frame, bboxes[i])

        ts = max(int(round(i * 1000.0 / fps)) + prev_ts + 1, prev_ts + 1)
        prev_ts = ts
        res = handle.model.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=patch), ts)

        out = _landmarks(res)
        if out is None:
            continue
        img_norm, lm_wld[i], vis[i] = out
        lm_img[i, :, 0] = ox + img_norm[:, 0] * s
        lm_img[i, :, 1] = oy + img_norm[:, 1] * s
        lm_img[i, :, 2] = img_norm[:, 2] * s
        found[i] = True
    handle.ultimo_ts = prev_ts
    cap.release()

    lm_img, lm_wld, vis = lm_img[:n], lm_wld[:n], vis[:n]
    out_mask = np.asarray(mask[:n], dtype=bool) & found[:n]
    if not out_mask.any():
        raise RuntimeError(f"MediaPipe non ha trovato nessuna posa in {n} frame "
                           f"di {Path(video_path).name}")

    kpts_h36m, scores = coco_to_h36m(to_coco(lm_img[..., :2], "blazepose33"),
                                     to_coco(vis, "blazepose33"))
    pose3d = coco_to_h36m_3d(to_coco(lm_wld, "blazepose33"))
    native = dict(native_format="blazepose33", lm_world=lm_wld, lm_img=lm_img,
                  visibility=vis)
    return pose3d, kpts_h36m, scores, out_mask, native
