# Percorsi di dati, pesi e repository esterni; all'import aggiunge PoseFormerV2 a sys.path.

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

VIDEOS_DIR = PROJECT_ROOT / "videos"
CACHE_DIR = PROJECT_ROOT / "cache"
KPTS3D_DIR = PROJECT_ROOT / "3dkeypoints"
FIT3D_DIR = PROJECT_ROOT / "fit3d dataset" / "train"
EVAL_DIR = PROJECT_ROOT / "evaluation"

YOLO_SUBJECT = PROJECT_ROOT / "yolo26s.pt"
YOLO_POSE = PROJECT_ROOT / "yolo26x-pose.pt"
BLAZEPOSE_TASK = PROJECT_ROOT / "pose3d" / "pose_landmarker_heavy.task"
BLAZEPOSE_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                 "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task")

POSEFORMER_ROOT = PROJECT_ROOT / "PoseFormerV2"
DEMO_ROOT = POSEFORMER_ROOT / "demo"
CFG_HRNET = DEMO_ROOT / "lib" / "hrnet" / "experiments" / "w48_384x288_adam_lr1e-3.yaml"
CKPT_HRNET = DEMO_ROOT / "lib" / "checkpoint" / "pose_hrnet_w48_384x288.pth"
CKPT_POSEFORMER = POSEFORMER_ROOT / "checkpoint" / "27_243_45.2.bin"

for _p in (POSEFORMER_ROOT, DEMO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

PEAR_ROOT = PROJECT_ROOT / "PEAR"
PEAR_CFG = PEAR_ROOT / "configs" / "infer.yaml"
PEAR_FLAME_DIR = PEAR_ROOT / "assets" / "FLAME"
PEAR_SMPLX_DIR = PEAR_ROOT / "assets" / "SMPLX"
J14_REGRESSOR = PEAR_SMPLX_DIR / "SMPLX_to_J14.pkl"

PEAR_BODY_MODELS = {
    PEAR_SMPLX_DIR / "SMPLX_NEUTRAL_2020.npz":
        "SMPLX_NEUTRAL_2020.npz da https://smpl-x.is.tue.mpg.de/download.php",
    PEAR_SMPLX_DIR / "flame_generic_model.pkl":
        "generic_model.pkl di FLAME2020 da https://flame.is.tue.mpg.de/download.php, "
        "rinominato",
    PEAR_FLAME_DIR / "FLAME2020" / "generic_model.pkl":
        "generic_model.pkl di FLAME2020 da https://flame.is.tue.mpg.de/download.php",
}


# Restituisce `path` se esiste, altrimenti solleva un errore leggibile.
def require(path, what):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{what} non trovato: {path}")
    return path


# Restituisce il modello BlazePose, o spiega come scaricarlo se manca.
def require_blazepose():
    if BLAZEPOSE_TASK.exists():
        return BLAZEPOSE_TASK
    raise FileNotFoundError(
        f"modello BlazePose non trovato: {BLAZEPOSE_TASK}\n"
        f"scaricalo con:\n  curl -L -o \"{BLAZEPOSE_TASK}\" {BLAZEPOSE_URL}")


# Controlla che ci siano i file del modello del corpo richiesti da PEAR.
def require_pear_assets():
    require(PEAR_CFG, "config di PEAR")
    mancanti = [p for p in PEAR_BODY_MODELS if not p.exists()]
    if mancanti:
        righe = "\n".join(f"  - {p.relative_to(PROJECT_ROOT)}\n"
                          f"      {PEAR_BODY_MODELS[p]}" for p in mancanti)
        raise FileNotFoundError(
            f"mancano {len(mancanti)} file del modello del corpo di PEAR:\n{righe}")
