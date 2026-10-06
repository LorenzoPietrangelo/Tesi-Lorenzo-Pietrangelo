# Forma canonica delle pose: assi della camera, scheletro H36M-17 e bacino nell'origine.

from types import SimpleNamespace

import numpy as np

from shared import npz

BORDO = 243 // 2


# Mask che esclude i frame in cui la finestra del lifter è completata replicando i bordi.
def senza_bordi(mask):
    mask = np.array(mask, dtype=bool)
    mask[:BORDO] = False
    mask[max(len(mask) - BORDO, 0):] = False
    return mask


# Legge un file di posa 3D e porta il bacino nell'origine in ogni frame.
def carica(path):
    d = npz.load_3d(path)
    mask = senza_bordi(d.mask) if d.lifter else d.mask
    return SimpleNamespace(
        xyz=d.pose3d - d.pose3d[:, :1],
        kpts=d.kpts_h36m,
        mask=mask,
        detector=d.detector,
        lifter=d.lifter,
        units=d.units,
        scala_nota=d.units == "m",
    )
