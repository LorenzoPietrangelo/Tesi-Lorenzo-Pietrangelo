# Metriche sui 12 giunti valutati: MPJPE, angoli, variazione delle ossa e riproiezione 2D.

import numpy as np

from shared.skeleton import JOINTS_DIRECT

PROTOCOLLI = ("root", "centroide", "nmpjpe", "pa")

OSSA_LUNGHE = [(1, 2), (2, 3), (4, 5), (5, 6),
               (14, 15), (15, 16), (11, 12), (12, 13)]

ANCHE = (1, 4)
SPALLE = (11, 14)
TRONCO = (ANCHE, SPALLE)

ANGOLI = {
    "ginocchio dx": ((2, 1), (2, 3)),
    "ginocchio sx": ((5, 4), (5, 6)),
    "anca dx": (TRONCO, (1, 2)),
    "anca sx": (TRONCO, (4, 5)),
    "gomito dx": ((15, 14), (15, 16)),
    "gomito sx": ((12, 11), (12, 13)),
}


# Maschera booleana dei frame, tutta vera se non indicata.
def _maschera(mask, n):
    return np.ones(n, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)


# Allinea i punti P su Q con la similitudine ottimale (Procuste), applicandola a P o ad `applica_a`.
def umeyama(P, Q, applica_a=None):
    mP, mQ = P.mean(0), Q.mean(0)
    P0, Q0 = P - mP, Q - mQ
    U, S, Vt = np.linalg.svd(P0.T @ Q0 / len(P))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    s = (S[:2].sum() + d * S[2]) / ((P0 ** 2).sum() / len(P))
    X = P0 if applica_a is None else applica_a - mP
    return s * X @ R.T + mQ


# Errore medio per giunto in mm dopo aver rimosso ciò che prevede il protocollo.
def mpjpe(pred, gt, mask=None, protocollo="root"):
    if protocollo not in PROTOCOLLI:
        raise ValueError(f"protocollo sconosciuto: {protocollo!r}, "
                         f"attesi {PROTOCOLLI}")
    mask = _maschera(mask, len(pred))
    if not mask.any():
        raise ValueError("nessun frame valido nella maschera")
    p = np.asarray(pred, dtype=np.float64)[mask][:, JOINTS_DIRECT]
    g = np.asarray(gt, dtype=np.float64)[mask][:, JOINTS_DIRECT]

    if protocollo == "centroide":
        p = p - p.mean(1, keepdims=True)
        g = g - g.mean(1, keepdims=True)
    elif protocollo == "nmpjpe":
        p = p - p.mean(1, keepdims=True)
        g = g - g.mean(1, keepdims=True)
        p = p * ((p * g).sum() / (p * p).sum())
    elif protocollo == "pa":
        p = np.stack([umeyama(a, b) for a, b in zip(p, g)])

    return 1000 * np.linalg.norm(p - g, axis=-1).mean(0)


# Coefficiente di variazione medio delle 8 ossa lunghe, in percentuale.
def cv_ossa(pose, mask=None):
    p = np.asarray(pose)[_maschera(mask, len(pose))]
    cv = []
    for a, b in OSSA_LUNGHE:
        v = np.linalg.norm(p[:, a] - p[:, b], axis=-1)
        cv.append(v.std() / v.mean() if v.mean() > 0 else np.nan)
    return float(100 * np.nanmean(cv))


# Posizione di un giunto, o punto medio di più giunti, in ogni frame.
def _punto(pose, giunti):
    return pose[:, list(np.atleast_1d(giunti))].mean(axis=1)


# Vettore di un segmento, dal primo punto al secondo.
def _segmento(pose, seg):
    return _punto(pose, seg[1]) - _punto(pose, seg[0])


# Angolo in gradi fra due segmenti, ciascuno dato come (origine, fine).
def angolo(pose, seg1, seg2):
    v1 = _segmento(pose, seg1)
    v2 = _segmento(pose, seg2)
    cos = np.einsum("ij,ij->i", v1, v2) / (np.linalg.norm(v1, axis=1) *
                                           np.linalg.norm(v2, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# MAE, bias e correlazione di ogni angolo articolare rispetto al ground truth.
def errore_angoli(pred, gt, mask=None):
    mask = _maschera(mask, len(pred))
    out = {}
    for nome, (seg1, seg2) in ANGOLI.items():
        ap = angolo(pred, seg1, seg2)[mask]
        ag = angolo(gt, seg1, seg2)[mask]
        corr = (float(np.corrcoef(ap, ag)[0, 1]) if ap.std() > 0 and ag.std() > 0
                else np.nan)
        out[nome] = (float(np.abs(ap - ag).mean()), float((ap - ag).mean()), corr)
    return out


# Errore mediano di riproiezione in pixel sui 12 giunti valutati.
def errore_2d(kpts, uv, mask=None):
    mask = _maschera(mask, len(kpts))
    return float(np.median(np.linalg.norm(
        np.asarray(kpts)[mask][:, JOINTS_DIRECT] -
        np.asarray(uv)[mask][:, JOINTS_DIRECT], axis=-1)))
