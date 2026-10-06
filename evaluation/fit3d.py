# Ground truth di Fit3D: giunti, ripetizioni, calibrazione, box comuni e sincronizzazione con i video.

import json
from types import SimpleNamespace

import numpy as np

import paths
from shared.skeleton import JOINTS_DIRECT

RISOLUZIONE = 900
FPS = 50.0

FIT3D_A_H36M = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10,
                         11, 12, 13, 14, 15, 16])

MARGINE = 1.25
ASPETTO = 288 / 384

OFFSETS = (0, 1, -1, 2, -2, 3, -3)


# Legge un file JSON del dataset, con un errore chiaro se manca.
def _json(path, cosa):
    with open(paths.require(path, cosa)) as f:
        return json.load(f)


# Soggetti presenti nel dataset.
def soggetti():
    return sorted(p.name for p in paths.FIT3D_DIR.iterdir() if p.is_dir())


# Esercizi disponibili per un soggetto.
def esercizi(soggetto):
    return sorted(p.stem for p in
                  (paths.FIT3D_DIR / soggetto / "joints3d_25").glob("*.json"))


# Camere disponibili per un soggetto.
def camere(soggetto):
    return sorted(p.name for p in
                  (paths.FIT3D_DIR / soggetto / "camera_parameters").iterdir())


# Percorso del video di una sequenza.
def video(soggetto, esercizio, cam):
    return paths.FIT3D_DIR / soggetto / "videos" / cam / f"{esercizio}.mp4"


# Giunti 3D in metri e coordinate mondo, in H36M-17 (Fit3D mette prima la gamba sinistra).
def giunti(soggetto, esercizio):
    d = _json(paths.FIT3D_DIR / soggetto / "joints3d_25" / f"{esercizio}.json",
              "giunti 3D di fit3d")
    return np.asarray(d["joints3d_25"], dtype=np.float32)[:, :17][:, FIT3D_A_H36M]


# Frame del ground truth che separano le ripetizioni di un esercizio.
def ripetizioni(soggetto, esercizio):
    d = _json(paths.FIT3D_DIR / soggetto / "rep_ann.json",
              "annotazione delle ripetizioni di fit3d")
    return [int(f) for f in d[esercizio]]


# Estrinseche e intrinseche, con distorsione, di una camera.
def camera(soggetto, esercizio, cam):
    d = _json(paths.FIT3D_DIR / soggetto / "camera_parameters" / cam /
              f"{esercizio}.json", "parametri di camera di fit3d")
    wd = d["intrinsics_w_distortion"]
    return dict(R=np.asarray(d["extrinsics"]["R"], dtype=np.float64),
                T=np.asarray(d["extrinsics"]["T"], dtype=np.float64).reshape(3),
                f=np.asarray(wd["f"], dtype=np.float64).reshape(2),
                c=np.asarray(wd["c"], dtype=np.float64).reshape(2),
                k=np.asarray(wd["k"], dtype=np.float64).reshape(3),
                p=np.asarray(wd["p"], dtype=np.float64).reshape(2))


# Porta i punti dalle coordinate mondo a quelle della camera.
def in_camera(g, cam):
    return (g - cam["T"]) @ cam["R"].T


# Proietta punti in coordinate camera nei pixel, con il modello di distorsione di Fit3D.
def proietta(gc, cam):
    x = gc[..., :2] / gc[..., 2:3]
    p = cam["p"][::-1]
    r2 = (x ** 2).sum(-1)
    radiale = 1 + cam["k"][0] * r2 + cam["k"][1] * r2 ** 2 + cam["k"][2] * r2 ** 3
    tangente = x @ p
    xx = x * (tangente + radiale)[..., None] + r2[..., None] * p
    return xx * cam["f"] + cam["c"]


# Box comuni dalla proiezione del ground truth, allargate di MARGINE e portate a 3:4.
def bbox(soggetto, esercizio, cam):
    par = camera(soggetto, esercizio, cam)
    uv = proietta(in_camera(giunti(soggetto, esercizio), par), par)

    x1, y1 = uv[:, :, 0].min(1), uv[:, :, 1].min(1)
    x2, y2 = uv[:, :, 0].max(1), uv[:, :, 1].max(1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w, h = (x2 - x1) * MARGINE, (y2 - y1) * MARGINE

    w = np.maximum(w, h * ASPETTO)
    h = np.maximum(h, w / ASPETTO)

    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return out.astype(np.float32), np.ones(len(out), dtype=bool)


# Sfasamento in frame fra video e ground truth che minimizza l'errore di riproiezione 2D.
def trova_offset(kpts_h36m, soggetto, esercizio, cam):
    par = camera(soggetto, esercizio, cam)
    uv = proietta(in_camera(giunti(soggetto, esercizio), par), par)
    migliore = (None, np.inf)
    for off in OFFSETS:
        a, b = max(off, 0), max(-off, 0)
        n = min(len(kpts_h36m) - a, len(uv) - b)
        if n < 10:
            continue
        e = float(np.median(np.linalg.norm(
            kpts_h36m[a:a + n, JOINTS_DIRECT] - uv[b:b + n, JOINTS_DIRECT],
            axis=-1)))
        if e < migliore[1]:
            migliore = (off, e)
    if migliore[0] is None:
        raise RuntimeError(f"nessun offset confrontabile per "
                           f"{soggetto}/{esercizio}/{cam}")
    return migliore


# Sincronizza una posa con il ground truth in coordinate camera, bacino nell'origine.
def allinea(posa, soggetto, esercizio, cam):
    offset, _ = trova_offset(posa.kpts, soggetto, esercizio, cam)
    par = camera(soggetto, esercizio, cam)
    gc = in_camera(giunti(soggetto, esercizio), par).astype(np.float32)

    a, b = max(offset, 0), max(-offset, 0)
    n = min(len(posa.xyz) - a, len(gc) - b)
    tagliata = SimpleNamespace(**{**vars(posa),
                                  "xyz": posa.xyz[a:a + n],
                                  "kpts": posa.kpts[a:a + n],
                                  "mask": posa.mask[a:a + n]})
    gt = gc[b:b + n]
    info = dict(offset=offset, n=n, inizio_gt=b, uv=proietta(gt, par))
    return tagliata, gt - gt[:, :1], info
