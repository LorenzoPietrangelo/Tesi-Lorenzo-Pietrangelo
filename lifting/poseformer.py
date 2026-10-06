# Lifting 2D -> 3D con PoseFormerV2 su finestre di 243 frame; uscita in metri.

from types import SimpleNamespace

import numpy as np
import torch

import paths
from shared import npz
from shared.skeleton import H36M_LEFT, H36M_RIGHT, coco_to_h36m
from shared.wrapper import Handle

from common.model_poseformer import PoseTransformerV2

LIFTER = "poseformer_v2"
ARGS = SimpleNamespace(frames=243, number_of_kept_frames=27,
                       number_of_kept_coeffs=27, embed_dim_ratio=32, depth=4)
BATCH = 64


# Costruisce PoseFormerV2 e carica il checkpoint 27_243_45.2.bin.
def load(device="cuda"):
    paths.require(paths.CKPT_POSEFORMER, "checkpoint PoseFormerV2")
    model = PoseTransformerV2(num_frame=ARGS.frames, num_joints=17, in_chans=2,
                              num_heads=8, mlp_ratio=2., qkv_bias=True,
                              qk_scale=None, drop_path_rate=0.1, args=ARGS)

    sd = torch.load(paths.CKPT_POSEFORMER, map_location="cpu", weights_only=False)
    sd = sd.get("model_pos", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    return Handle(model.to(device).eval(), device=device)


# Finestre di 243 frame centrate su ogni frame, ripetendo i bordi, a blocchi di BATCH.
def _windows(seq):
    pad = ARGS.frames // 2
    padded = np.pad(seq, ((pad, pad), (0, 0), (0, 0)), mode="edge")
    for s in range(0, len(seq), BATCH):
        e = min(s + BATCH, len(seq))
        yield np.stack([padded[i:i + ARGS.frames] for i in range(s, e)])


# Stima il frame centrale di ogni finestra, mediata con quella dell'ingresso specchiato.
@torch.no_grad()
def _forward(handle, x):
    t = torch.from_numpy(x).float().to(handle.device)
    out = handle.model(t)

    lr, rl = H36M_LEFT + H36M_RIGHT, H36M_RIGHT + H36M_LEFT
    tf = t.clone()
    tf[..., 0] *= -1
    tf[:, :, lr] = tf[:, :, rl]
    of = handle.model(tf)
    of[..., 0] *= -1
    of[:, :, lr] = of[:, :, rl]
    out = (out + of) * 0.5

    out[:, :, 0] = 0
    return out[:, 0].cpu().numpy()


# Normalizza i keypoint H36M-17 come PoseFormerV2 e li solleva in 3D.
def lift(handle, kpts_h36m, width, height):
    seq = kpts_h36m.astype(np.float32) / width * 2 - [1, height / width]
    return np.concatenate([_forward(handle, w) for w in _windows(seq)], axis=0)


# Converte un file di keypoint 2D COCO-17 in un file di posa 3D.
def lift_npz(handle, kpts_2d_path, out_path):
    d = npz.load_2d(kpts_2d_path)
    kpts_h36m, scores_h36m = coco_to_h36m(d.keypoints, d.scores)
    pose3d = lift(handle, kpts_h36m, int(d.meta["width"]), int(d.meta["height"]))
    return npz.save_3d(out_path, pose3d, kpts_h36m, scores_h36m, d.mask, d.meta,
                       d.detector, units="m", lifter=LIFTER, **d.extra)
