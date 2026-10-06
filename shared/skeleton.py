# Conversioni fra scheletri: dal formato nativo di ogni modello a COCO-17 e poi a Human3.6M-17.

import numpy as np

H36M_NAMES = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]
H36M_LEFT = [4, 5, 6, 11, 12, 13]
H36M_RIGHT = [1, 2, 3, 14, 15, 16]

JOINTS_DIRECT = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]

NATIVI = {
    "blazepose33": (33, [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16,
                         23, 24, 25, 26, 27, 28]),
    "coco_wholebody133": (133, list(range(17))),
    "smplx145": (145, [134, 126, 137, 125, 136,
                       16, 17, 18, 19, 20, 21,
                       1, 2, 4, 5, 7, 8]),
}

H36M_COCO_ORDER = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
COCO_ORDER = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


# Seleziona i 17 giunti COCO-17 da uno scheletro nativo (coordinate o confidenze).
def to_coco(punti, formato):
    k, idx = NATIVI[formato]
    punti = np.asarray(punti, dtype=np.float32)
    if punti.shape[1] != k:
        raise ValueError(f"attesi {k} punti {formato}, trovato {punti.shape}")
    return punti[:, idx]


# Riordina i giunti COCO-17 in H36M-17 e costruisce i cinque mancanti (Tabella A.2).
def _coco_a_h36m(coco):
    out = np.zeros_like(coco)
    out[:, H36M_COCO_ORDER] = coco[:, COCO_ORDER]

    shoulders = coco[:, [5, 6]].mean(axis=1)
    hips = coco[:, [11, 12]].mean(axis=1)

    out[:, 0] = hips
    out[:, 8] = shoulders + (coco[:, 0] - shoulders) / 3
    out[:, 7] = coco[:, [5, 6, 11, 12]].mean(axis=1)
    out[:, 10] = coco[:, 1:5].mean(axis=1)
    out[:, 9] -= (out[:, 9] - shoulders) / 4
    return out


# Converte keypoint 2D COCO-17 e confidenze in H36M-17, con le stesse regole della posa 3D.
def coco_to_h36m(keypoints, scores):
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.shape[1:] != (17, 2):
        raise ValueError(f"attesi (N, 17, 2) COCO-17, trovato {keypoints.shape}")

    conf = np.zeros_like(scores)
    conf[:, H36M_COCO_ORDER] = scores[:, COCO_ORDER]
    conf[:, 0] = scores[:, [11, 12]].mean(axis=1)
    conf[:, 8] = scores[:, [5, 6]].mean(axis=1)
    conf[:, 7] = conf[:, [0, 8]].mean(axis=1)
    conf[:, 10] = scores[:, 1:5].mean(axis=1)
    return _coco_a_h36m(keypoints), conf


# Converte una posa 3D COCO-17 in H36M-17.
def coco_to_h36m_3d(coco):
    coco = np.asarray(coco, dtype=np.float32)
    if coco.shape[1:] != (17, 3):
        raise ValueError(f"attesi (N, 17, 3), trovato {coco.shape}")
    return _coco_a_h36m(coco)
