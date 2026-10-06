# Metriche sulla rappresentazione parametrica di PEAR: errore per vertice ed errore geodetico contro lo SMPL-X di Fit3D.

import json
import time

import numpy as np

import paths
import pipeline
from evaluation import fit3d, metrics
from evaluation.raccolta import salva_csv, scegli, sequenza

CSV_VERTICI = paths.EVAL_DIR / "errore_vertici.csv"
CSV_GEODETICO = paths.EVAL_DIR / "errore_geodetico.csv"
PASSO = 5

GIUNTI_B12 = (1, 2, 4, 5, 7, 8, 16, 17, 18, 19, 20, 21)

NOMI_BODY_POSE = ["anca sx", "anca dx", "colonna 1", "ginocchio sx",
                  "ginocchio dx", "colonna 2", "caviglia sx", "caviglia dx",
                  "colonna 3", "piede sx", "piede dx", "collo", "clavicola sx",
                  "clavicola dx", "testa", "spalla sx", "spalla dx",
                  "gomito sx", "gomito dx", "polso sx", "polso dx"]
BROOK12 = {"anca sx", "anca dx", "ginocchio sx", "ginocchio dx", "caviglia sx",
           "caviglia dx", "spalla sx", "spalla dx", "gomito sx", "gomito dx",
           "polso sx", "polso dx"}

INTESTAZIONE_VERTICI = ["sistema", "soggetto", "esercizio", "camera",
                        "protocollo", "insieme", "errore_mm", "n_frame"]
INTESTAZIONE_GEODETICO = ["sistema", "soggetto", "esercizio", "camera",
                          "giunto", "brook12", "errore_deg"]


# Il modello SMPL-X neutro, con i soli campi usati da forward_smplx.
def carica_smplx():
    d = np.load(paths.PEAR_SMPLX_DIR / "SMPLX_NEUTRAL_2020.npz", allow_pickle=True)
    return {k: np.asarray(d[k], dtype=np.float64) for k in
            ("v_template", "shapedirs", "posedirs", "J_regressor",
             "kintree_table", "weights")}


# Linear blend skinning di SMPL-X: vertici e giunti cinematici di un frame.
def forward_smplx(m, betas, expr, full_pose, transl):
    v = (m["v_template"] + m["shapedirs"][:, :, :len(betas)] @ betas
         + m["shapedirs"][:, :, 300:300 + len(expr)] @ expr)
    J = m["J_regressor"] @ v
    v = v + m["posedirs"] @ (full_pose[1:] - np.eye(3)).reshape(-1)

    parents = m["kintree_table"][0].astype(int)
    parents[0] = -1
    n = len(parents)
    rel = J.copy()
    rel[1:] -= J[parents[1:]]
    T = np.zeros((n, 4, 4))
    T[:, :3, :3] = full_pose
    T[:, :3, 3] = rel
    T[:, 3, 3] = 1.0
    G = [T[0]]
    for i in range(1, n):
        G.append(G[parents[i]] @ T[i])
    G = np.stack(G)
    giunti = G[:, :3, 3].copy()

    Jh = np.concatenate([J, np.zeros((n, 1))], axis=1)
    G[:, :3, 3] -= np.einsum("nij,nj->ni", G, Jh)[:, :3]
    Tv = np.einsum("vn,nij->vij", m["weights"], G)
    vh = np.concatenate([v, np.ones((len(v), 1))], axis=1)
    vertici = np.einsum("vij,vj->vi", Tv, vh)[:, :3]
    return vertici + transl, giunti + transl


# Insiemi di vertici, dal giunto dominante nei pesi di skinning.
def insiemi(m):
    dom = m["weights"].argmax(1)
    return {"brook12": np.isin(dom, GIUNTI_B12),
            "corpo": dom < 22,
            "intera": np.ones(len(dom), dtype=bool)}


# Parametri SMPL-X del riferimento di Fit3D per un esercizio.
def _riferimento(soggetto, esercizio):
    with open(paths.require(paths.FIT3D_DIR / soggetto / "smplx" /
                            f"{esercizio}.json", "smplx di fit3d")) as f:
        return {k: np.asarray(v, dtype=np.float64) for k, v in json.load(f).items()}


# Frame corrispondenti di PEAR e del riferimento, dopo la sincronizzazione.
def _frame(d, s, e, c, n_rif, passo=1):
    off, _ = fit3d.trova_offset(d["kpts_h36m"], s, e, c)
    a, b = max(off, 0), max(-off, 0)
    n = min(len(d["smplx_pose"]) - a, n_rif - b)
    k = np.arange(0, n, passo)
    k = k[np.asarray(d["mask"], dtype=bool)[a + k]]
    return a + k, b + k


# Errore medio per vertice in mm, con centraggio o similitudine stimati sui vertici `sel`.
def _pve(P, Q, protocollo, radice_p, radice_q, sel):
    p, q = P.astype(np.float64), Q.astype(np.float64)
    if protocollo == "root":
        p, q = p - radice_p[:, None], q - radice_q[:, None]
    elif protocollo == "centroide":
        p = p - p[:, sel].mean(1, keepdims=True)
        q = q - q[:, sel].mean(1, keepdims=True)
    elif protocollo == "nmpjpe":
        p = p - p[:, sel].mean(1, keepdims=True)
        q = q - q[:, sel].mean(1, keepdims=True)
        p = p * ((p[:, sel] * q[:, sel]).sum() / (p[:, sel] * p[:, sel]).sum())
    elif protocollo == "pa":
        p = np.stack([metrics.umeyama(x[sel], y[sel], applica_a=x)
                      for x, y in zip(p, q)])
    return 1000 * np.linalg.norm(p - q, axis=-1).mean(0)


# Righe di errore_vertici.csv per una sequenza, un frame ogni PASSO.
def righe_vertici(handle, m, regioni, s, e, c):
    from pose3d import pear

    d = dict(np.load(pipeline.file_3d(sequenza(s, e, c), "pear")))
    rif = _riferimento(s, e)
    pose_r = np.concatenate([rif[k] for k in
                             ("global_orient", "body_pose", "jaw_pose",
                              "leye_pose", "reye_pose", "left_hand_pose",
                              "right_hand_pose")], axis=1)
    ip, ir = _frame(d, s, e, c, len(rif["transl"]), PASSO)
    V = len(m["v_template"])

    Vp = np.concatenate([pear.mesh(handle, d, ip[i:i + pear.BATCH])[1][:, :V]
                         for i in range(0, len(ip), pear.BATCH)])
    Rp = d["lm_world"][ip, 0]
    Vq, Rq = zip(*(forward_smplx(m, rif["betas"][b], rif["expression"][b],
                                 pose_r[b], rif["transl"][b]) for b in ir))
    par = fit3d.camera(s, e, c)
    Vq = fit3d.in_camera(np.stack(Vq), par)
    Rq = fit3d.in_camera(np.stack(Rq)[:, :1], par)[:, 0]

    for prot in metrics.PROTOCOLLI:
        for nome, sel in regioni.items():
            err = _pve(Vp, Vq, prot, Rp, Rq, sel)[sel].mean()
            yield ["pear", s, e, c, prot, nome, round(float(err), 3), len(ip)]


# Angolo in gradi fra coppie di rotazioni.
def geodetica(A, B):
    tr = np.einsum("...ij,...ij->...", A, B)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


# Righe di errore_geodetico.csv: rotazioni relative al padre dei 21 giunti di body_pose.
def righe_geodetico(s, e, c):
    d = dict(np.load(pipeline.file_3d(sequenza(s, e, c), "pear")))
    rif = _riferimento(s, e)["body_pose"].astype(np.float32)
    ip, ir = _frame(d, s, e, c, len(rif))
    err = geodetica(d["smplx_pose"][ip, 1:22], rif[ir]).mean(0)
    for nome, v in zip(NOMI_BODY_POSE, err):
        yield ["pear", s, e, c, nome, nome in BROOK12, round(float(v), 3)]


# Calcola errore per vertice ed errore geodetico di PEAR e salva i due CSV.
def main(prova=False, device=None):
    from pose3d import pear

    seq = scegli(prova)
    m = carica_smplx()
    regioni = insiemi(m)
    vertici, geo = [], []
    print(f"{len(seq)} sequenze, errore per vertice su un frame ogni {PASSO}")
    with pear.load(device=pipeline.pick_device(device)) as h:
        for i, (s, e, c) in enumerate(seq, 1):
            t0 = time.time()
            vertici += righe_vertici(h, m, regioni, s, e, c)
            geo += righe_geodetico(s, e, c)
            print(f"  [{i:2d}/{len(seq)}] {s}_{e}_{c} in {time.time() - t0:.1f} s",
                  flush=True)
    salva_csv(CSV_VERTICI, INTESTAZIONE_VERTICI, vertici)
    salva_csv(CSV_GEODETICO, INTESTAZIONE_GEODETICO, geo)
