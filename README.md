# Stima della posa 3D negli esercizi con bilanciere

Codice della tesi di laurea triennale in Informatica *A Comparative Analysis of
3D Human Pose Estimation via Computer Vision in Strength Lifting* (Lorenzo
Pietrangelo, relatore Prof. Juri Di Rocco, A.A. 2025-2026).

Il progetto rende confrontabili sei sistemi di stima monoculare della posa 3D,
appartenenti a tre paradigmi, e li valuta su squat e stacco da terra:

| Paradigma | Sistema | Uscita |
|---|---|---|
| due stadi (2D + lifting) | HRNet-w48 → PoseFormerV2 | metri |
| due stadi (2D + lifting) | ViTPose+ huge → PoseFormerV2 | metri |
| due stadi (2D + lifting) | YOLO26x-pose → PoseFormerV2 | metri |
| end-to-end a keypoint | BlazePose heavy (MediaPipe) | metri |
| end-to-end a keypoint | RTMW3D-x (rtmlib) | normalizzata |
| end-to-end parametrico | PEAR (mesh SMPL-X) | metri |

Tutti i sistemi ricevono le stesse bounding box e le loro uscite vengono
convertite nello stesso scheletro (Human3.6M-17) e negli stessi assi. La
valutazione usa il dataset [Fit3D](https://fit3d.imar.ro/) come riferimento: 8
soggetti, 2 esercizi, 4 telecamere, 64 sequenze.

## Struttura

```
main.py          interfaccia a riga di comando
pipeline.py      elenco dei sistemi e stadi: bounding box -> 2D -> lifting -> 3D
subject.py       bounding box del soggetto su un video qualsiasi (YOLO26s + BoT-SORT)
paths.py         percorsi del progetto e controlli sui file mancanti
pose2d/          HRNet, ViTPose+, YOLO26-pose
pose3d/          BlazePose, RTMW3D, PEAR
lifting/         PoseFormerV2
shared/          contratto dei wrapper, formato dei file, conversioni fra scheletri
evaluation/      riferimento Fit3D, forma canonica, metriche, raccolta e metriche parametriche
risultati/       CSV della tesi, un file per metrica
PoseFormerV2/    repository di terze parti (MIT), da cui vengono HRNet e il lifter
PEAR/            repository di terze parti (Apache 2.0)
```

Ogni modello espone `load()` e `run()` (vedi `shared/wrapper.py`); la pipeline
salva il risultato di ogni stadio e lo riusa nelle esecuzioni successive.

### Come le uscite diventano confrontabili

1. **Regione di interesse comune.** Tutti i sistemi ricevono le stesse box:
   su Fit3D sono la proiezione del ground truth (`evaluation/fit3d.py`), su un
   video qualsiasi le calcola `subject.py`.
2. **Scheletro comune** (`shared/skeleton.py`). Ogni formato nativo
   (COCO-17, BlazePose-33, COCO-WholeBody-133, SMPL-X di PEAR) viene ridotto a
   COCO-17 per selezione di indici e poi convertito in Human3.6M-17: nel ramo
   2D + lifting sull'ingresso di PoseFormerV2, nei sistemi 3D diretti
   sull'uscita del modello. In entrambi i casi i dodici giunti valutati sono
   solo riordinati e i cinque mancanti sono costruiti con le formule della
   Tabella A.2 della tesi.
3. **Giunti di PEAR** (`pose3d/pear.py`). PEAR stima una mesh, non keypoint:
   forma del corpo e del volto e scale di mani e testa vengono fissate alla
   mediana sulla sequenza, poi tutti i dodici giunti valutati vengono dal
   regressore J14 applicato alla superficie.
4. **Forma canonica** (`evaluation/canon.py`). Assi della camera, bacino
   nell'origine, scala dichiarata (`units`). Il ground truth viene portato
   nella stessa forma con la calibrazione di Fit3D e sincronizzato con il
   video. Per i sistemi con il lifter sono esclusi dalle metriche i 121 frame
   iniziali e finali, dove la finestra di 243 frame è completata replicando i
   bordi.
5. **Metriche** (`evaluation/metrics.py`) sui 12 giunti stimati da tutti i
   sistemi (anche, ginocchia, caviglie, spalle, gomiti, polsi): MPJPE ancorato
   al bacino e al baricentro, N-MPJPE, PA-MPJPE, errore angolare (ginocchio fra
   coscia e tibia, anca fra tronco e coscia, con il tronco dal punto medio
   delle anche al punto medio delle spalle), coefficiente di variazione delle
   ossa ed errore 2D. Gli angoli sono riportati anche per ripetizione, dalle
   annotazioni di Fit3D. Per PEAR `evaluation/parametriche.py` calcola anche
   l'errore per vertice e l'errore geodetico sulle rotazioni articolari.

## Installazione

Ambiente verificato: Windows, Python 3.11, GPU NVIDIA RTX 2060 con CUDA 12.1.

```bash
python -m venv .venv
.venv\Scripts\activate

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install --no-deps rtmlib==0.0.16          # dichiara due varianti di OpenCV in conflitto
pip install --no-build-isolation chumpy==0.70 # solo per PEAR
```

PEAR richiede anche `pytorch3d`, che su Windows non ha pacchetti
precompilati. PEAR ne usa solo parti in puro PyTorch, quindi basta copiare la
cartella `pytorch3d/` dei sorgenti (v0.7.9) in `site-packages` e aggiungere
un file `pytorch3d/_C.py` con le due costanti lette all'import:

```python
EPS = 1e-6
MAX_UINT = 4294967295
```

### Pesi e dati da procurare

| Elemento | Dove va | Provenienza |
|---|---|---|
| HRNet-w48 384x288 | `PoseFormerV2/demo/lib/checkpoint/pose_hrnet_w48_384x288.pth` | link nel README di PoseFormerV2 |
| PoseFormerV2 27/243 | `PoseFormerV2/checkpoint/27_243_45.2.bin` | link nel README di PoseFormerV2 |
| BlazePose heavy | `pose3d/pose_landmarker_heavy.task` | [MediaPipe](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task) |
| SMPL-X neutro 2020 | `PEAR/assets/SMPLX/SMPLX_NEUTRAL_2020.npz` | [smpl-x.is.tue.mpg.de](https://smpl-x.is.tue.mpg.de/), licenza di ricerca |
| FLAME 2020 | `PEAR/assets/SMPLX/flame_generic_model.pkl` e `PEAR/assets/FLAME/FLAME2020/generic_model.pkl` (stesso file) | [flame.is.tue.mpg.de](https://flame.is.tue.mpg.de/), licenza di ricerca |
| Fit3D, split train | `fit3d dataset/train/` | [fit3d.imar.ro](https://fit3d.imar.ro/), previa registrazione |

YOLO26 (`yolo26s.pt`, `yolo26x-pose.pt`), ViTPose+, RTMW3D e PEAR scaricano i
propri pesi al primo uso. `python main.py modelli` indica, per ogni sistema,
che cosa manca ancora.

PEAR richiede una GPU NVIDIA: la sua testa di predizione crea un tensore su
`"cuda"` indipendentemente dal device richiesto.

## Uso

### Un sistema su un video

```bash
python main.py modelli                                  # stato dei sistemi
python main.py esegui videos/squat.mp4 -m hrnet_w48_384x288
python main.py esegui -m pear                           # tutti i video in videos/
```

Le bounding box vengono da `subject.py` e sono condivise da tutti i sistemi;
il risultato è in `3dkeypoints/<video>_<sistema>.npz`.

### Valutazione su Fit3D

I comandi vanno eseguiti in quest'ordine, perché `parametriche` usa i file di
PEAR prodotti da `raccolta`:

```bash
python main.py raccolta          # bounding box dal ground truth, sei sistemi e LAGANN, un CSV per metrica
python main.py parametriche      # errore per vertice ed errore geodetico di PEAR
```

`raccolta` accetta `--prova` (una sola sequenza), `--modelli`, `--soggetti`,
`--camere`, `--device`, `--force` e `--solo-csv` (ricalcola i CSV dai file
già prodotti).

I due comandi scrivono in `evaluation/` un CSV per ogni metrica della Tabella
4.2 della tesi, con gli stessi nomi, le stesse colonne e lo stesso ordine delle
righe dei file in `risultati/`, da cui sono state calcolate le tabelle: il
confronto si fa file per file (`evaluation/pa_mpjpe.csv` con
`risultati/pa_mpjpe.csv`, e così via). In più `raccolta` scrive
`angoli_ripetizioni.csv`, con le metriche angolari di ogni ripetizione.

| File | Metrica | Comando |
|---|---|---|
| `mpjpe_bacino.csv`, `mpjpe_baricentro.csv`, `n_mpjpe.csv`, `pa_mpjpe.csv` | MPJPE ai quattro livelli di allineamento, per giunto | `raccolta` |
| `errore_angolare.csv`, `bias_angolare.csv`, `correlazione_angolare.csv` | metriche angolari, per articolazione e lato | `raccolta` |
| `cv_ossa.csv`, `errore_2d.csv`, `quota_rilevatore.csv` | CV delle ossa, errore 2D, quota del rilevatore | `raccolta` |
| `errore_vertici.csv`, `errore_geodetico.csv` | errore per vertice ed errore geodetico di PEAR | `parametriche` |

## Formato dei risultati

`3dkeypoints/<sequenza>_<sistema>.npz` (formato completo in `shared/npz.py`):

| Campo | Forma | Contenuto |
|---|---|---|
| `pose3d` | (N, 17, 3) | posa Human3.6M-17 negli assi della camera, unità in `units` |
| `kpts_h36m` | (N, 17, 2) | stessi giunti in pixel del frame |
| `scores` | (N, 17) | confidenza per giunto |
| `mask` | (N,) | frame in cui il soggetto è stato osservato |
| `units` | | `"m"` oppure `"normalized"` (solo forma, senza scala) |
| `detector`, `width`, `height`, `fps`, `n_frames` | | metadati |

Assi: x a destra, y in basso, z in profondità; l'origine è propria di ogni
sistema e la valutazione la porta sul bacino. Ordine dei giunti: Hip, RHip,
RKnee, RFoot, LHip, LKnee, LFoot, Spine, Thorax, Neck/Nose, Head, LShoulder,
LElbow, LWrist, RShoulder, RElbow, RWrist. Hip, Spine, Thorax, Neck/Nose e
Head sono ricavati dagli altri giunti; la valutazione usa i 12 giunti di anche,
ginocchia, caviglie, spalle, gomiti e polsi. I sistemi 3D diretti salvano anche
l'uscita nel proprio scheletro nativo (`native_format`, `lm_world`, `lm_img`);
PEAR salva inoltre i parametri SMPL-X e FLAME (`smplx_*`, `flame_*`, `pd_cam`),
con la forma già resa costante.

## Codice e modelli di terze parti

- [PoseFormerV2](https://github.com/QitaoZhao/PoseFormerV2) (MIT), con la demo
  di HRNet. In `PoseFormerV2/` ci sono solo i file usati; l'unica modifica è
  l'import relativo in `demo/lib/hrnet/lib/utils/inference.py`, che
  nell'originale registrava un package `utils` in conflitto con quello di PEAR.
- [PEAR](https://github.com/Pixel-Talk/PEAR) (Apache 2.0): in `PEAR/` ci sono
  solo i file usati per l'inferenza, non modificati.
- ViTPose+ tramite [transformers](https://github.com/huggingface/transformers),
  YOLO26 tramite [ultralytics](https://github.com/ultralytics/ultralytics)
  (AGPL-3.0), BlazePose tramite [MediaPipe](https://github.com/google-ai-edge/mediapipe),
  RTMW3D tramite [rtmlib](https://github.com/Tau-J/rtmlib).
- SMPL-X, FLAME e Fit3D sono soggetti alle rispettive licenze di ricerca e non
  sono inclusi.

## Licenza

Il codice del progetto è distribuito con licenza MIT (vedi `LICENSE`). Le
cartelle `PoseFormerV2/` e `PEAR/` restano sotto le proprie licenze (MIT e
Apache 2.0), incluse nei rispettivi file `LICENSE`.
