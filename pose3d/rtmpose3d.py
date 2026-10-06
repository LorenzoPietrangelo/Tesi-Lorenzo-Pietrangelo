# Posa 3D con RTMW3D-x tramite rtmlib e ONNX, in coordinate normalizzate e non in metri.

import cv2
import numpy as np
from rtmlib import RTMPose3d

from shared.skeleton import coco_to_h36m, coco_to_h36m_3d, to_coco
from shared.wrapper import Handle

MODEL_URL = ("https://huggingface.co/Soykaf/RTMW3D-x/resolve/main/onnx/"
             "rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx")

INPUT_SIZE = (288, 384)
Z_RANGE = 2.1744869
PADDING = 1.25
ASPECT = INPUT_SIZE[0] / INPUT_SIZE[1]


# Handle che rilascia la sessione di onnxruntime.
class _Onnx(Handle):
    # Rilascia la sessione.
    def close(self):
        self.model.session = None


# Carica il modello ONNX, scaricato in ~/.cache/rtmlib al primo uso.
def load(device="cuda"):
    net = RTMPose3d(MODEL_URL, model_input_size=INPUT_SIZE,
                    backend="onnxruntime", device=device)
    return _Onnx(net, device=device, units="normalized")


# Pixel del frame per pixel del ritaglio, per riportare la posa alla scala del frame.
def _scala_crop(box):
    x1, y1, x2, y2 = (float(v) for v in box)
    w, h = abs(x2 - x1) * PADDING, abs(y2 - y1) * PADDING
    if w > h * ASPECT:
        h = w / ASPECT
    return h / INPUT_SIZE[1]


# Stima i 133 keypoint COCO-WholeBody di ogni frame e li converte in H36M-17.
def run(handle, video_path, bboxes, mask):
    n = len(bboxes)
    wb3d = np.zeros((n, 133, 3), dtype=np.float32)
    wb2d = np.zeros((n, 133, 2), dtype=np.float32)
    conf = np.zeros((n, 133), dtype=np.float32)

    cap = cv2.VideoCapture(str(video_path))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            break
        box = [float(v) for v in bboxes[i]]
        _, sc, simcc, kp2d = handle.model(frame, bboxes=[box])

        wb3d[i] = (simcc[0] / INPUT_SIZE[1] - 0.5) * 2.0 * Z_RANGE * _scala_crop(box)
        wb2d[i] = kp2d[0]
        conf[i] = sc[0]
    cap.release()

    wb3d, wb2d, conf = wb3d[:n], wb2d[:n], conf[:n]
    kpts_h36m, scores = coco_to_h36m(to_coco(wb2d, "coco_wholebody133"),
                                     to_coco(conf, "coco_wholebody133"))
    pose3d = coco_to_h36m_3d(to_coco(wb3d, "coco_wholebody133"))
    native = dict(native_format="coco_wholebody133", lm_world=wb3d,
                  lm_img=wb2d, visibility=conf)
    return pose3d, kpts_h36m, scores, np.asarray(mask[:n], dtype=bool), native
