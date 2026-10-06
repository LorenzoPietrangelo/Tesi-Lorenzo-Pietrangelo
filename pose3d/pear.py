# Posa 3D con PEAR: forma del corpo costante nella sequenza, giunti valutati tutti dal regressore J14.

import contextlib
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import paths
from shared.skeleton import coco_to_h36m, coco_to_h36m_3d, to_coco
from shared.wrapper import Handle

HF_REPO = "BestWJH/PEAR_models"
HF_FILE = "pear_model.pt"

CROP = 256
RATIO = 1.25
FOCAL = 24.0
BATCH = 24

J14_ANCHE_CAVIGLIE = {3: 1, 2: 2, 5: 7, 0: 8}

IDENTITA = ("smplx_shape", "smplx_hand_scale", "smplx_head_scale", "flame_shape")


# Rimette in inspect e numpy i nomi rimossi che chumpy usa ancora.
def _compat():
    import inspect
    import warnings
    if not hasattr(inspect, "getargspec"):
        inspect.getargspec = inspect.getfullargspec

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        for nome, tipo in (("bool", bool), ("int", int), ("float", float),
                           ("complex", complex), ("object", object),
                           ("str", str), ("unicode", str),
                           ("nan", float("nan")), ("inf", float("inf"))):
            try:
                getattr(np, nome)
            except AttributeError:
                setattr(np, nome, tipo)


# Esegue il blocco con la directory corrente in PEAR/, dove il modello cerca gli asset.
@contextlib.contextmanager
def _cwd_pear():
    prima = os.getcwd()
    os.chdir(paths.PEAR_ROOT)
    try:
        yield
    finally:
        os.chdir(prima)


# Carica rete, modello del corpo SMPL-X/FLAME e regressore J14; richiede una GPU.
def load(device="cuda"):
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "PEAR richiede una GPU NVIDIA: PEAR/models/smplx/smplx_head.py "
            "usa .to(\"cuda\") ignorando il device richiesto")
    paths.require_pear_assets()

    _compat()
    if str(paths.PEAR_ROOT) not in sys.path:
        sys.path.insert(0, str(paths.PEAR_ROOT))
    from huggingface_hub import hf_hub_download
    from models.modules.ehm import EHM_v2
    from models.pipeline.ehm_pipeline import Ehm_Pipeline
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(paths.PEAR_CFG))
    ckpt = torch.load(hf_hub_download(repo_id=HF_REPO, filename=HF_FILE),
                      map_location="cpu", weights_only=True)
    with _cwd_pear():
        model = Ehm_Pipeline(cfg)
        model.backbone.load_state_dict(ckpt["backbone"], strict=False)
        model.head.load_state_dict(ckpt["head"], strict=False)
        ehm = EHM_v2(str(paths.PEAR_FLAME_DIR), str(paths.PEAR_SMPLX_DIR))

    with open(paths.J14_REGRESSOR, "rb") as f:
        j14 = np.asarray(pickle.load(f, encoding="latin1"), dtype=np.float32)

    h = Handle(model.to(device).eval(), device=device, units="m", ratio=RATIO,
               forma="mediana")
    h.ehm = ehm.to(device).eval()
    h.j14 = j14
    return h


# Centro e lato del ritaglio quadrato, come sanitize_bbox e process_bbox di PEAR.
def _finestra(box, fw, fh):
    x1, y1, x2, y2 = map(float, box)
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2 = min(fw - 1.0, x1 + max(0.0, w - 1.0))
    y2 = min(fh - 1.0, y1 + max(0.0, h - 1.0))
    if not (w * h > 0 and x2 > x1 and y2 > y1):
        return (0.0, 0.0), 0.0
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0), max(x2 - x1, y2 - y1) * RATIO


# Ritaglio RGB 256x256 attorno alla box, con centro e lato; None se la box è degenere.
def _crop(frame_bgr, box):
    fh, fw = frame_bgr.shape[:2]
    (cx, cy), s = _finestra(box, fw, fh)
    if s == 0:
        return None, (cx, cy), s

    k = CROP / s
    trans = np.array([[k, 0.0, CROP / 2.0 - k * cx],
                      [0.0, k, CROP / 2.0 - k * cy]], dtype=np.float32)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    patch = cv2.warpAffine(rgb.astype(np.float32), trans, (CROP, CROP),
                           flags=cv2.INTER_LINEAR)
    x = (patch / 255.0).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(x, dtype=np.float32), (cx, cy), s


# Proietta i giunti nei pixel del frame con la camera predetta da PEAR.
def _project(joints, cam, center, s):
    tx, ty, tz = (float(v) for v in cam[:3, 3])
    depth = joints[:, 2] + tz
    depth = np.where(np.abs(depth) < 1e-6, np.nan, depth)

    f = FOCAL * s / 2.0
    out = np.empty((len(joints), 2), dtype=np.float32)
    out[:, 0] = center[0] + f * (joints[:, 0] - tx) / depth
    out[:, 1] = center[1] + f * (joints[:, 1] - ty) / depth
    return out


# Parametri SMPL-X e FLAME di un frame su CPU; le pose sono matrici di rotazione (1 + 21 + 15 + 15).
@torch.no_grad()
def _infer(handle, x):
    out = handle.model(torch.from_numpy(x).to(handle.device))
    b, f = out["body_param"], out["flame_param"]

    # Primo elemento del batch come array float32.
    def a(t):
        return t[0].detach().cpu().numpy().astype(np.float32)

    return dict(
        smplx_pose=np.concatenate([a(b["global_pose"]), a(b["body_pose"]),
                                   a(b["left_hand_pose"]),
                                   a(b["right_hand_pose"])]),
        smplx_shape=a(b["shape"]), smplx_exp=a(b["exp"]),
        smplx_hand_scale=a(b["hand_scale"]), smplx_head_scale=a(b["head_scale"]),
        flame_eye_pose=a(f["eye_pose_params"]), flame_pose=a(f["pose_params"]),
        flame_jaw=a(f["jaw_params"]), flame_eyelid=a(f["eyelid_params"]),
        flame_exp=a(f["expression_params"]), flame_shape=a(f["shape_params"]),
        pd_cam=a(out["pd_cam"]),
    )


# Raccoglie i parametri dei frame in array (N, ...), con zeri dove il frame manca.
def _stack(params, n):
    campi = next((p for p in params if p is not None), None)
    if campi is None:
        return {}
    out = {}
    for k, v in campi.items():
        a = np.zeros((n, *np.shape(v)), dtype=np.float32)
        for i, p in enumerate(params):
            if p is not None:
                a[i] = p[k]
        out[k] = a
    return out


# Fissa forma del corpo e del volto e scale di mani e testa alla mediana sui frame validi.
def forma_costante(params, mask):
    for campo in IDENTITA:
        stima = np.median(params[campo][mask], axis=0)
        params[campo] = np.repeat(stima[None], len(params[campo]), axis=0)
    return params


# Giunti (145) e vertici della mesh dei frame `idx`, dai parametri salvati.
@torch.no_grad()
def mesh(handle, params, idx):
    idx = np.atleast_1d(idx)

    # Campo dei parametri dei frame richiesti, come tensore sul device.
    def t(key):
        return torch.from_numpy(np.asarray(params[key][idx])).to(handle.device)

    pose = t("smplx_pose")
    body = dict(global_pose=pose[:, :1], body_pose=pose[:, 1:22],
                left_hand_pose=pose[:, 22:37], right_hand_pose=pose[:, 37:52],
                hand_scale=t("smplx_hand_scale"), head_scale=t("smplx_head_scale"),
                shape=t("smplx_shape"), exp=t("smplx_exp"),
                eye_pose=None, jaw_pose=None, joints_offset=None)
    flame = dict(eye_pose_params=t("flame_eye_pose"), pose_params=t("flame_pose"),
                 jaw_params=t("flame_jaw"), eyelid_params=t("flame_eyelid"),
                 expression_params=t("flame_exp"), shape_params=t("flame_shape"))
    out = handle.ehm(body, flame)
    return (out["joints"].cpu().numpy().astype(np.float32),
            out["vertices"].cpu().numpy().astype(np.float32))


# Stima i parametri di ogni frame, fissa la forma e ricava i giunti con il regressore J14.
def run(handle, video_path, bboxes, mask):
    n = len(bboxes)
    params, finestre = [], []
    found = np.zeros(n, dtype=bool)

    cap = cv2.VideoCapture(str(video_path))
    for i in range(n):
        ok, frame = cap.read()
        if not ok:
            n = i
            break
        x, center, s = _crop(frame, bboxes[i])
        finestre.append((center, s))
        if x is None:
            params.append(None)
            continue
        params.append(_infer(handle, x))
        found[i] = True
    cap.release()

    in_mask = np.asarray(mask[:n], dtype=bool)
    out_mask = in_mask & found[:n]
    if not out_mask.any():
        causa = ("il soggetto non è mai stato rilevato" if not in_mask.any()
                 else "tutte le bounding box sono degeneri")
        raise RuntimeError(f"nessun frame utilizzabile su {n} di "
                           f"{Path(video_path).name}: {causa}")

    params = forma_costante(_stack(params[:n], n), out_mask)

    nativi = np.zeros((n, 145, 3), dtype=np.float32)
    giunti = np.zeros((n, 145, 3), dtype=np.float32)
    kpts_nativi = np.zeros((n, 145, 2), dtype=np.float32)
    kpts = np.zeros((n, 145, 2), dtype=np.float32)
    righe = list(J14_ANCHE_CAVIGLIE)
    indici = list(J14_ANCHE_CAVIGLIE.values())
    nv = handle.j14.shape[1]

    validi = np.flatnonzero(found[:n])
    for i0 in range(0, len(validi), BATCH):
        idx = validi[i0:i0 + BATCH]
        j, v = mesh(handle, params, idx)
        nativi[idx] = j
        giunti[idx] = j
        giunti[idx[:, None], indici] = np.einsum(
            "jv,bvc->bjc", handle.j14[righe], v[:, :nv])
        for i in idx:
            center, s = finestre[i]
            kpts_nativi[i] = _project(nativi[i], params["pd_cam"][i], center, s)
            kpts[i] = _project(giunti[i], params["pd_cam"][i], center, s)

    kpts_h36m, scores = coco_to_h36m(to_coco(kpts, "smplx145"),
                                     np.ones((n, 17), dtype=np.float32))
    pose3d = coco_to_h36m_3d(to_coco(giunti, "smplx145"))
    native = dict(native_format="smplx145", lm_world=nativi,
                  lm_img=kpts_nativi, **params)
    return pose3d, kpts_h36m, scores, out_mask, native
