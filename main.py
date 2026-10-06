# Interfaccia a riga di comando: esecuzione dei sistemi sui video e valutazione su Fit3D.

import argparse
from pathlib import Path

import paths
import pipeline


# Stampa i sei sistemi e cosa manca per eseguirli.
def modelli(_args):
    for tag, m in pipeline.MODELS.items():
        ramo = "due stadi" if m.lifting else "end-to-end"
        mancano = pipeline.missing(m)
        stato = "pronto" if not mancano else "manca: " + ", ".join(mancano)
        print(f"{tag:20s} {m.label:20s} {ramo:14s} {stato}")


# Trova un video dal percorso o dal nome di un file in videos/.
def _video(nome):
    for p in (Path(nome), paths.VIDEOS_DIR / nome):
        if p.is_file():
            return p
    raise SystemExit(f"video non trovato: {nome}")


# Esegue un sistema sui video indicati, o su tutti quelli in videos/.
def esegui(args):
    import numpy as np

    m = pipeline.MODELS[args.modello]
    mancano = pipeline.missing(m)
    if mancano:
        raise SystemExit(f"{m.tag} non eseguibile, manca: {', '.join(mancano)}")
    video = [_video(v) for v in args.video] or pipeline.find_videos()
    if not video:
        raise SystemExit("nessun video indicato e nessun video in videos/")

    for v in video:
        print(f"\n== {v.name} con {m.label}")
        out = pipeline.esegui_video(v, m.tag, args.device, args.force)
        d = np.load(out)
        print(f"   {len(d['pose3d'])} frame, soggetto osservato in "
              f"{int(d['mask'].sum())}, unità: {d['units']}\n   -> {out}")


# Esegue i sistemi su Fit3D e scrive un CSV per metrica, LAGANN compreso.
def raccolta(args):
    from evaluation import raccolta
    raccolta.main(args.prova, args.modelli, args.soggetti, args.camere,
                  args.force, args.solo_csv, args.device)


# Scrive errore_vertici.csv ed errore_geodetico.csv (solo PEAR).
def parametriche(args):
    from evaluation import parametriche
    parametriche.main(args.prova, args.device)


# Legge la riga di comando ed esegue il comando scelto.
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="main.py", description="Stima della posa 3D con sei sistemi resi "
        "confrontabili, e loro valutazione su fit3d.")
    sub = ap.add_subparsers(dest="comando", required=True, metavar="COMANDO")
    device = dict(default=None, help="cuda o cpu (default: cuda se disponibile)")
    prova = dict(action="store_true", help="una sola sequenza (s03, squat, "
                 "camera 50591643), per collaudare la catena")

    p = sub.add_parser("modelli", help="sistemi disponibili e cosa manca")
    p.set_defaults(fn=modelli)

    p = sub.add_parser("esegui", help="posa 3D di uno o più video con un sistema")
    p.add_argument("video", nargs="*", help="percorso o nome di un file in "
                   "videos/ (default: tutti i video in videos/)")
    p.add_argument("-m", "--modello", default=pipeline.DEFAULT_MODEL,
                   choices=list(pipeline.MODELS), metavar="MODELLO",
                   help=f"uno di: {', '.join(pipeline.MODELS)} "
                        f"(default {pipeline.DEFAULT_MODEL})")
    p.add_argument("--device", **device)
    p.add_argument("--force", action="store_true",
                   help="ricalcola anche gli stadi già in cache")
    p.set_defaults(fn=esegui)

    p = sub.add_parser("raccolta", help="esegue i sei sistemi su fit3d e "
                       "scrive un CSV per metrica, come in risultati/")
    p.add_argument("--prova", **prova)
    p.add_argument("--modelli", nargs="+", choices=list(pipeline.MODELS),
                   metavar="MODELLO", help="solo questi sistemi")
    p.add_argument("--soggetti", nargs="+", help="solo questi soggetti (es. s03)")
    p.add_argument("--camere", nargs="+", help="solo queste camere")
    p.add_argument("--device", **device)
    p.add_argument("--force", action="store_true",
                   help="ricalcola anche quanto è già in cache")
    p.add_argument("--solo-csv", action="store_true",
                   help="non esegue i sistemi, ricalcola i CSV dai file esistenti")
    p.set_defaults(fn=raccolta)

    p = sub.add_parser("parametriche", help="errore per vertice ed errore "
                       "geodetico di PEAR contro lo SMPL-X di fit3d")
    p.add_argument("--prova", **prova)
    p.add_argument("--device", **device)
    p.set_defaults(fn=parametriche)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
