# Esegue i sei sistemi sulle sequenze di Fit3D e scrive un CSV per ogni metrica della Tabella 4.2 della tesi.

import csv
from collections import defaultdict
from types import SimpleNamespace

import numpy as np

import paths
import pipeline
from evaluation import canon, fit3d, metrics
from shared import npz

ESERCIZI = ("deadlift", "squat")
SISTEMI = tuple(pipeline.MODELS)
LAGANN = "lagann"
ORDINE_SISTEMI = (LAGANN,) + SISTEMI
PROVA = ("s03", "squat", "50591643")

NOMI_GIUNTI = ["RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
               "LSho", "LElb", "LWri", "RSho", "RElb", "RWri"]
ORDINE_GIUNTI = ["RHip", "LHip", "RKnee", "LKnee", "RFoot", "LFoot",
                 "RSho", "LSho", "RElb", "LElb", "RWri", "LWri"]
GRUPPI = {"RHip": "anche", "LHip": "anche", "RKnee": "ginocchia",
          "LKnee": "ginocchia", "RFoot": "caviglie", "LFoot": "caviglie",
          "RSho": "spalle", "LSho": "spalle", "RElb": "gomiti",
          "LElb": "gomiti", "RWri": "polsi", "LWri": "polsi"}

SEQ = ["soggetto", "esercizio", "camera"]
FILE_MPJPE = {"root": "mpjpe_bacino", "centroide": "mpjpe_baricentro",
              "nmpjpe": "n_mpjpe", "pa": "pa_mpjpe"}
FILE_ANGOLI = ("errore_angolare", "bias_angolare", "correlazione_angolare")
CIFRE_ANGOLI = (3, 3, 4)

INTESTAZIONI = {
    **{f: ["sistema", "ramo", *SEQ, "giunto", "gruppo", "errore_mm", "n_frame"]
       for f in FILE_MPJPE.values()},
    "errore_angolare": ["sistema", "ramo", *SEQ, "articolazione", "lato",
                        "errore_deg"],
    "bias_angolare": ["sistema", "ramo", *SEQ, "articolazione", "lato",
                      "bias_deg"],
    "correlazione_angolare": ["sistema", "ramo", *SEQ, "articolazione", "lato",
                              "correlazione"],
    "cv_ossa": ["sistema", "ramo", *SEQ, "cv_percento"],
    "errore_2d": ["sistema", "ramo", *SEQ, "errore_px"],
    "quota_rilevatore": ["sistema", *SEQ, "pa_mpjpe_mm", "pa_mpjpe_lagann_mm",
                         "quota_percento"],
    "angoli_ripetizioni": ["sistema", "ramo", *SEQ, "ripetizione",
                           "articolazione", "lato", "errore_deg", "bias_deg",
                           "correlazione", "n_frame"],
}


# Triple (soggetto, esercizio, camera) di squat e stacco con un video disponibile.
def sequenze(soggetti=None, camere=None):
    out = []
    for s in soggetti or fit3d.soggetti():
        disponibili = fit3d.esercizi(s)
        for e in ESERCIZI:
            if e not in disponibili:
                continue
            for c in camere or fit3d.camere(s):
                if fit3d.video(s, e, c).exists():
                    out.append((s, e, c))
    return out


# Converte una tripla di Fit3D nella Sequenza della pipeline, con la provenienza.
def sequenza(s, e, c):
    video = fit3d.video(s, e, c)
    return pipeline.Sequenza(f"{s}_{e}_{c}", video,
                             dict(soggetto=s, esercizio=e, camera=c,
                                  bbox_da="ground_truth", sorgente=str(video)))


# Nome del ramo di un sistema, come compare nei CSV.
def ramo(tag):
    if tag == LAGANN or pipeline.MODELS[tag].lifting:
        return "due stadi"
    return "end-to-end"


# Scrive un CSV con la sua intestazione.
def salva_csv(path, intestazione, righe):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(intestazione)
        w.writerows(righe)
    print(f"[csv] {len(righe)} righe -> {path}")


# Calcola dal ground truth le bounding box che mancano in cache.
def prepara_bbox(seq, force=False):
    meta = dict(width=fit3d.RISOLUZIONE, height=fit3d.RISOLUZIONE, fps=fit3d.FPS)
    fatte = 0
    for s, e, c in seq:
        out = pipeline.file_bboxes(sequenza(s, e, c))
        if out.exists() and not force:
            continue
        npz.save_bboxes(out, *fit3d.bbox(s, e, c), meta)
        fatte += 1
    print(f"[bbox] {fatte} nuove, {len(seq) - fatte} riusate")


# Prepara le bounding box ed esegue i sistemi scelti su tutte le sequenze.
def raccogli(seq, modelli=SISTEMI, device=None, force=False):
    print(f"{len(seq)} sequenze, {len(modelli)} sistemi")
    prepara_bbox(seq, force)
    for tag in modelli:
        print(f"\n== {pipeline.MODELS[tag].label}", flush=True)
        pipeline.esegui(tag, [sequenza(*t) for t in seq], device, force)


# Per ogni file di posa esistente restituisce posa e ground truth già sincronizzati.
def allineate(seq):
    for s, e, c in seq:
        for tag in SISTEMI:
            f = pipeline.file_3d(sequenza(s, e, c), tag)
            if not f.exists():
                continue
            try:
                p, gt, info = fit3d.allinea(canon.carica(f), s, e, c)
            except Exception as exc:
                print(f"  [salto] {f.name}: {type(exc).__name__}: {exc}")
                continue
            yield tag, (s, e, c), p, gt, info


# LAGANN: PoseFormerV2 applicato al ground truth proiettato, nella forma delle pose allineate.
def lagann(seq, device=None):
    from lifting import poseformer

    with poseformer.load(device=pipeline.pick_device(device)) as h:
        for s, e, c in seq:
            par = fit3d.camera(s, e, c)
            gc = fit3d.in_camera(fit3d.giunti(s, e), par).astype(np.float32)
            uv = fit3d.proietta(gc, par).astype(np.float32)
            pose = poseformer.lift(h, uv, fit3d.RISOLUZIONE, fit3d.RISOLUZIONE)
            p = SimpleNamespace(xyz=pose - pose[:, :1], kpts=uv, scala_nota=True,
                                mask=canon.senza_bordi(np.ones(len(pose))))
            yield LAGANN, (s, e, c), p, gc - gc[:, :1], dict(uv=uv, inizio_gt=0)


# Valore arrotondato, o stringa vuota se non definito (correlazione con serie costante).
def _arrotonda(v, cifre):
    return "" if np.isnan(v) else round(float(v), cifre)


# Righe degli angoli ristretti a ogni ripetizione annotata in Fit3D.
def righe_ripetizioni(tag, ramo_, sec, p, gt, inizio_gt):
    confini = np.asarray(fit3d.ripetizioni(sec[0], sec[1])) - inizio_gt
    for k, (a, b) in enumerate(zip(confini[:-1], confini[1:]), 1):
        m = np.zeros(len(p.xyz), dtype=bool)
        m[max(a, 0):max(b, 0)] = True
        m &= p.mask
        if m.sum() < 2:
            continue
        for nome, valori in metrics.errore_angoli(p.xyz, gt, m).items():
            articolazione, lato = nome.rsplit(" ", 1)
            yield [tag, ramo_, *sec, k, articolazione, lato,
                   *map(_arrotonda, valori, CIFRE_ANGOLI), int(m.sum())]


# Aggiunge alle tabelle le righe di una posa allineata, una tabella per metrica.
def aggiungi(tab, tag, sec, p, gt, info):
    r = ramo(tag)
    n = int(p.mask.sum())
    for prot, nome in FILE_MPJPE.items():
        if prot in ("root", "centroide") and not p.scala_nota:
            continue
        err = dict(zip(NOMI_GIUNTI, metrics.mpjpe(p.xyz, gt, p.mask, prot)))
        tab[nome] += [[tag, r, *sec, g, GRUPPI[g], round(float(err[g]), 3), n]
                      for g in ORDINE_GIUNTI]

    for nome, valori in metrics.errore_angoli(p.xyz, gt, p.mask).items():
        articolazione, lato = nome.rsplit(" ", 1)
        for f, v, cifre in zip(FILE_ANGOLI, valori, CIFRE_ANGOLI):
            tab[f].append([tag, r, *sec, articolazione, lato, _arrotonda(v, cifre)])

    tab["cv_ossa"].append([tag, r, *sec, round(metrics.cv_ossa(p.xyz, p.mask), 3)])
    if tag != LAGANN:
        tab["errore_2d"].append([tag, r, *sec, round(
            metrics.errore_2d(p.kpts, info["uv"], p.mask), 2)])
    tab["angoli_ripetizioni"] += righe_ripetizioni(tag, r, sec, p, gt,
                                                   info["inizio_gt"])


# Quota del rilevatore (eq. 4.4) di ogni sequenza dei sistemi a due stadi, dal PA-MPJPE medio.
def righe_quota(righe_pa):
    acc = defaultdict(list)
    for tag, _, s, e, c, _, _, err, _ in righe_pa:
        acc[(tag, s, e, c)].append(err)
    pa = {k: float(np.mean(v)) for k, v in acc.items()}

    out = []
    for (tag, s, e, c), v in pa.items():
        lim = pa.get((LAGANN, s, e, c))
        if tag == LAGANN or not pipeline.MODELS[tag].lifting or lim is None:
            continue
        out.append([tag, s, e, c, round(v, 3), round(lim, 3),
                    round(100 * (v - lim) / v, 2)])
    return out


# Ordina le righe per sistema e poi per sequenza, come nei CSV della tesi.
def _ordina(righe, inizio_seq):
    return sorted(righe, key=lambda r: (ORDINE_SISTEMI.index(r[0]),
                                        *map(str, r[inizio_seq:inizio_seq + 3])))


# Calcola le metriche dei sistemi e di LAGANN e scrive un CSV per ogni metrica.
def scrivi_csv(seq, device=None):
    tab = defaultdict(list)
    for tag, sec, p, gt, info in allineate(seq):
        aggiungi(tab, tag, sec, p, gt, info)
    print("\nLAGANN (PoseFormerV2 sul ground truth proiettato)", flush=True)
    for tag, sec, p, gt, info in lagann(seq, device):
        aggiungi(tab, tag, sec, p, gt, info)
    tab["quota_rilevatore"] = righe_quota(tab["pa_mpjpe"])

    for nome, intestazione in INTESTAZIONI.items():
        inizio_seq = intestazione.index("soggetto")
        salva_csv(paths.EVAL_DIR / f"{nome}.csv", intestazione,
                  _ordina(tab[nome], inizio_seq))


# Sequenze da valutare: quella di prova oppure quelle selezionate.
def scegli(prova=False, soggetti=None, camere=None):
    paths.require(paths.FIT3D_DIR, "dataset fit3d (vedi README.md)")
    seq = [PROVA] if prova else sequenze(soggetti, camere)
    if not seq:
        raise SystemExit(f"nessuna sequenza di squat o stacco in {paths.FIT3D_DIR}")
    return seq


# Esegue i sistemi (salvo solo_csv) e scrive i CSV delle metriche.
def main(prova=False, modelli=None, soggetti=None, camere=None, force=False,
         solo_csv=False, device=None):
    seq = scegli(prova, soggetti, camere)
    if not solo_csv:
        raccogli(seq, modelli or SISTEMI, device, force)
    scrivi_csv(seq, device)
