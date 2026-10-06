# Lettura e scrittura dei file .npz della pipeline: bounding box, keypoint 2D e pose 3D.

from pathlib import Path
from types import SimpleNamespace

import numpy as np

META = ("width", "height", "fps", "n_frames")
CAMPI_2D = ("keypoints", "scores", "mask", "format", "detector") + META


# Salva i campi e i metadati del video, con n_frames pari ai frame scritti.
def _scrivi(path, n, meta, **campi):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **campi, **{k: meta[k] for k in META[:-1]}, n_frames=n)
    return path


# Estrae i metadati del video da un file caricato.
def _meta(d):
    return {k: d[k] for k in META}


# Salva le bounding box del soggetto e la mask dei frame osservati.
def save_bboxes(path, bboxes, mask, meta):
    return _scrivi(path, len(bboxes), meta,
                   bboxes=np.asarray(bboxes, dtype=np.float32),
                   mask=np.asarray(mask, dtype=bool))


# Legge un file di bounding box: restituisce bboxes, mask e metadati.
def load_bboxes(path):
    d = np.load(path)
    return d["bboxes"], d["mask"], _meta(d)


# Salva i keypoint 2D COCO-17, azzerando le confidenze dei frame non osservati.
def save_2d(path, keypoints, scores, mask, meta, detector, **extra):
    mask = np.asarray(mask, dtype=bool)
    scores = np.array(scores, dtype=np.float32)
    scores[~mask] = 0.0
    return _scrivi(path, len(keypoints), meta,
                   keypoints=np.asarray(keypoints, dtype=np.float32),
                   scores=scores, mask=mask, format="coco17", detector=detector,
                   **extra)


# Legge un file di keypoint 2D; in `extra` finiscono parametri e provenienza.
def load_2d(path):
    d = np.load(path)
    if str(d["format"]) != "coco17":
        raise ValueError(f"atteso format='coco17' in {path}, "
                         f"trovato {str(d['format'])!r}")
    return SimpleNamespace(keypoints=d["keypoints"], scores=d["scores"],
                           mask=d["mask"].astype(bool),
                           detector=str(d["detector"]), meta=_meta(d),
                           extra={k: d[k] for k in d.files if k not in CAMPI_2D})


# Salva una posa 3D in H36M-17 con i keypoint 2D e i metadati.
def save_3d(path, pose3d, kpts_h36m, scores, mask, meta, detector, units,
            **extra):
    return _scrivi(path, len(pose3d), meta,
                   pose3d=np.asarray(pose3d, dtype=np.float32),
                   kpts_h36m=np.asarray(kpts_h36m, dtype=np.float32),
                   scores=np.asarray(scores, dtype=np.float32),
                   mask=np.asarray(mask, dtype=bool), format="h36m17",
                   units=units, detector=detector, **extra)


# Legge un file di posa 3D, rifiutando quelli della versione precedente.
def load_3d(path):
    d = np.load(path)
    if str(d["format"]) != "h36m17":
        raise ValueError(f"atteso format='h36m17' in {path}, "
                         f"trovato {str(d['format'])!r}")
    if "world" in d.files:
        raise ValueError(f"{path} è stato prodotto dalla versione precedente "
                         f"della pipeline, con altri assi: va rigenerato")
    return SimpleNamespace(pose3d=d["pose3d"], kpts_h36m=d["kpts_h36m"],
                           scores=d["scores"], mask=d["mask"].astype(bool),
                           units=str(d["units"]), detector=str(d["detector"]),
                           lifter=str(d["lifter"]) if "lifter" in d.files else None,
                           meta=_meta(d))
