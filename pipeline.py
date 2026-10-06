# Pipeline da video a posa 3D: elenco dei sei sistemi, nomi dei file e stadi con cache.

import importlib
import importlib.util
import time
from pathlib import Path
from typing import NamedTuple

import paths
from shared import npz


# Un sistema del confronto: wrapper, dipendenze e file da procurare.
class Model(NamedTuple):
    tag: str
    label: str
    module: str
    deps: tuple
    files: tuple = ()
    cuda: bool = False

    # True per i sistemi del ramo 2D + lifting.
    @property
    def lifting(self):
        return self.module.startswith("pose2d.")


_LIFTING = ("torch", "torch_dct", "timm", "einops")

MODELS = {m.tag: m for m in (
    Model("hrnet_w48_384x288", "HRNet-w48 384x288", "pose2d.hrnet",
          _LIFTING + ("yacs",), (paths.CKPT_HRNET, paths.CKPT_POSEFORMER)),
    Model("vitpose_plus_huge", "ViTPose+ huge", "pose2d.vitpose",
          _LIFTING + ("transformers",), (paths.CKPT_POSEFORMER,)),
    Model("yolo26x_pose", "YOLO26x-pose", "pose2d.yolo",
          _LIFTING + ("ultralytics",), (paths.CKPT_POSEFORMER,)),
    Model("blazepose_heavy", "BlazePose heavy", "pose3d.blazepose",
          ("mediapipe",), (paths.BLAZEPOSE_TASK,)),
    Model("rtmw3d_x", "RTMW3D-x", "pose3d.rtmpose3d",
          ("rtmlib", "onnxruntime")),
    Model("pear", "PEAR", "pose3d.pear",
          ("torch", "pytorch3d", "chumpy", "huggingface_hub", "omegaconf",
           "timm", "roma", "einops", "lightning"),
          tuple(paths.PEAR_BODY_MODELS), cuda=True),
)}
DEFAULT_MODEL = "blazepose_heavy"
VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv")


# Un video da elaborare, con il prefisso dei suoi file e la provenienza da salvare.
class Sequenza(NamedTuple):
    nome: str
    video: Path
    provenienza: dict


# File delle bounding box di una sequenza.
def file_bboxes(seq):
    return paths.CACHE_DIR / f"{seq.nome}_bboxes.npz"


# File dei keypoint 2D di una sequenza per un sistema.
def file_2d(seq, tag):
    return paths.CACHE_DIR / f"{seq.nome}_{tag}_2d.npz"


# File della posa 3D di una sequenza per un sistema.
def file_3d(seq, tag):
    return paths.KPTS3D_DIR / f"{seq.nome}_{tag}.npz"


# True se il pacchetto è importabile, senza importarlo.
def _installed(dep):
    try:
        return importlib.util.find_spec(dep) is not None
    except (ImportError, ValueError):
        return False


# True se torch vede una GPU CUDA.
def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# Elenca cosa manca per eseguire un sistema: pacchetti, file, GPU.
def missing(model):
    out = [f"pacchetto {d}" for d in model.deps if not _installed(d)]
    out += [f"file {Path(f).relative_to(paths.PROJECT_ROOT).as_posix()}"
            for f in model.files if not Path(f).exists()]
    if model.cuda and not _cuda():
        out.append("GPU NVIDIA con CUDA")
    return out


# Il device richiesto, altrimenti cuda se disponibile, altrimenti cpu.
def pick_device(device=None):
    if device is not None:
        return device
    return "cuda" if _cuda() else "cpu"


# Video presenti in videos/, in ordine alfabetico.
def find_videos():
    folder = paths.require(paths.VIDEOS_DIR, "cartella dei video")
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS)


# Esegue un sistema su sequenze con le box già in cache, saltando i file già prodotti.
def esegui(tag, sequenze, device=None, force=False):
    m = MODELS[tag]
    device = pick_device(device)
    uscita = file_2d if m.lifting else file_3d
    da_fare = [s for s in sequenze if force or not uscita(s, tag).exists()]

    if da_fare:
        wrapper = importlib.import_module(m.module)
        with wrapper.load(device=device) as h:
            for i, seq in enumerate(da_fare, 1):
                t0 = time.time()
                bboxes, mask, meta = npz.load_bboxes(file_bboxes(seq))
                out = wrapper.run(h, seq.video, bboxes, mask)
                if m.lifting:
                    kpts, scores, mask = out
                    npz.save_2d(file_2d(seq, tag), kpts, scores, mask, meta,
                                tag, **h.extra, **seq.provenienza)
                else:
                    pose3d, kpts_h36m, scores, mask, native = out
                    npz.save_3d(file_3d(seq, tag), pose3d, kpts_h36m, scores,
                                mask, meta, tag, h.units, **h.extra, **native,
                                **seq.provenienza)
                print(f"  [{tag}] {i}/{len(da_fare)} {seq.nome}: {len(mask)} "
                      f"frame in {time.time() - t0:.0f} s", flush=True)

    if m.lifting:
        _lifting(tag, sequenze, device, force)


# Solleva in 3D con PoseFormerV2 i keypoint 2D di un sistema.
def _lifting(tag, sequenze, device, force):
    da_fare = [s for s in sequenze if force or not file_3d(s, tag).exists()]
    if not da_fare:
        return
    from lifting import poseformer

    with poseformer.load(device=device) as h:
        for i, seq in enumerate(da_fare, 1):
            poseformer.lift_npz(h, file_2d(seq, tag), file_3d(seq, tag))
            print(f"  [lifting {tag}] {i}/{len(da_fare)} {seq.nome}", flush=True)


# Posa 3D di un video qualsiasi con le box di subject.py; restituisce il file prodotto.
def esegui_video(video, tag=DEFAULT_MODEL, device=None, force=False):
    video = paths.require(video, "video")
    seq = Sequenza(video.stem, video, dict(bbox_da="tracker", sorgente=str(video)))
    device = pick_device(device)

    if force or not file_bboxes(seq).exists():
        import subject
        bboxes, mask, meta = subject.extract_subject(video, device)
        npz.save_bboxes(file_bboxes(seq), bboxes, mask, meta)
    esegui(tag, [seq], device, force)
    return file_3d(seq, tag)
