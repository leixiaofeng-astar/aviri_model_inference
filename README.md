# AVIRI Visual-Impairment Fundus Classifier — Inference Guide

Self-contained inference pipeline for binary visual-impairment screening on
colour-fundus photos. Supports nine backbones:

| Flag                    | Architecture                                            | Input |
| ----------------------- | ------------------------------------------------------- | ----- |
| `--backbone vit_base`         | ViT-Base/16 ImageNet                               | 224   |
| `--backbone swin_v2`          | Swin V2 Base                                       | 256   |
| `--backbone retfound`         | RETFound CFP ViT-L/16                              | 224   |
| `--backbone retfound_mae`     | RETFound MAE CFP ViT-L/16 (drop_path=0.2, BICUBIC) | 224   |
| `--backbone resnet50`         | torchvision ResNet-50                              | 224   |
| `--backbone vit_large_mae`    | ImageNet-MAE ViT-L/16                              | 224   |
| `--backbone vit_base_mae`     | ImageNet-MAE ViT-B/16                              | 224   |
| `--backbone dino_vitbase16`   | ImageNet-DINO ViT-B/16                             | 224   |
| `--backbone retizero`         | RetiZero ViT-L/16 + LoRA rank 8                    | 224   |
| `--backbone visionfm`         | VisionFM ViT-B/16 + MLP head (last-4-block CLS)    | 224   |

Three uncertainty-estimation methods (`softmax`, `mc_dropout`, `edl`) — the
MC Dropout and EDL pathways are wired for the ViT/Swin/RETFound checkpoints
only; the newer foundation-model variants (`retizero`, `visionfm`,
`vit_large_mae`, `vit_base_mae`, `dino_vitbase16`) ship with softmax-only
checkpoints. Runs on GPU when available and falls back to CPU automatically.

The `visionfm` backbone uses VisionFM's retinal channel stats (mean/std from
`VisionFM-main/utils.py::get_stats("Fundus")`) and BICUBIC resize — all other
backbones use ImageNet mean/std and BILINEAR (except `retfound_mae`, which
uses BICUBIC to match its upstream MAE training pipeline).

`retfound_mae` is the same ViT-L/16 architecture and weight format as
`retfound`, but with `drop_path_rate=0.2` and BICUBIC resize (matching the
newer RETFound-MAE `main_finetune.py` defaults). It can be used with the same
checkpoint — e.g. the SEED-only softmax checkpoint — and typically yields a
slightly different operating point from `--backbone retfound` due to the
BICUBIC interpolation.

`retizero` loads fine-tuned LoRA ViT-L/16 checkpoints produced by
`RetiZero/finetune_retizero.py`; it does **not** require the RetiZero.pth
pretrained CLIP weights at inference, since the fine-tuned ``.pth`` file
supplies every tensor (encoder + LoRA adapters + classifier).

`visionfm` loads fine-tuned VisionFM checkpoints produced by
`VisionFM-main/finetune_visionfm_for_multiclass_classification.py`. These
checkpoints have a two-dict layout — `visionfm_state_dict` (backbone) and
`classifier_state_dict` (3-layer MLP head, DDP-wrapped) — handled
automatically by the loader.

---

## 1. Folder layout

```
model_inference/
├── infer_aviri.py            # inference entrypoint (this is what you run)
├── own_models_5.py           # backbone definitions (required for torch.load)
├── models_vit.py             # vendored RETFound ViT-L definition
├── visionfm_model.py         # vendored minimal VisionFM ViT-B + MLP head
├── fundus_preprocess.py      # fundus-disc crop preprocessor
├── requirements.txt          # Python package list
├── INFERENCE_GUIDE.md        # this file
└── models_weight/
    ├── vit_base/
    │   ├── aviri_class_model4vit_base_lr1e-05-AUC9428.pth          # softmax
    │   ├── aviri_class_model4vit_base_mc_dropout_lr1e-05-AUC9437.pth
    │   └── aviri_class_model4vit_base_edl_lr1e-05-AUC9464.pth
    ├── swin_v2/
    │   ├── aviri_class_model4swin_v2_lr1e-05-AUC9514.pth           # softmax
    │   ├── aviri_class_model4swin_v2_mc_dropout_lr1e-05-AUC9518.pth
    │   └── aviri_class_model4swin_v2_edl_lr1e-06-AUC9444.pth
    ├── retfound/
    │   ├── checkpoint-best_SEEDonly_AUC9497.pth                    # softmax
    │   ├── aviri_class_model4retfound_mc_dropout_lr5e-05-AUC9493.pth
    │   └── aviri_class_model4retfound_edl_lr5e-05-AUC9489.pth
    ├── resnet50/
    │   └── aviri_class_model4ResNet50_lr1e-05-AUC9454.pth          # softmax
    ├── vit_large_mae/
    │   └── aviri_class_model4vit_large_mae_lr3e-05-AUC9505.pth     # softmax
    ├── dino_vitbase16/
    │   └── aviri_class_model4dino_vitbase16_lr1e-06-AUC9472.pth    # softmax
    ├── retizero/
    │   └── retizero_lr1e-4_best-AUC9443.pth                        # softmax
    └── visionfm/
        └── checkpoint_best_finetune-AUC0.9484.pth                  # softmax
```

The checkpoints under `models_weight/` are the exact ones referenced in the
training commands and have already been staged here (the new-backbone
checkpoints are symlinks to their original locations so no bytes are
duplicated). The five newer backbones — `resnet50`, `vit_large_mae`,
`dino_vitbase16`, `retizero`, `visionfm` — are softmax-only (no MC-dropout or
EDL variants).

---

## 2. Environment setup

Requires Ubuntu, Python 3.10+ (3.12 recommended), and (optionally) a
CUDA-capable GPU.

```bash
cd /data/xiaofeng/HCSA/model_inference
python3.12 -m venv aviri_env
source aviri_env/bin/activate
```

Install PyTorch for your CUDA version (CPU-only also fine) from
<https://pytorch.org/get-started/locally/>, e.g.:

```bash
# CUDA 12.4
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
# CPU only
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Preparing your images

`--input_dir` accepts two layouts.

**With ground-truth labels (metrics are reported):**

```
input_dir/
├── normal/     # label 0 — healthy fundus images
│   ├── a.jpg
│   └── ...
└── vi/         # label 1 — visually impaired fundus images
    ├── b.png
    └── ...
```

**Prediction-only (no labels):**

```
input_dir/
├── img001.jpg
├── img002.png
└── ...
```

Accepted image extensions: `.png .jpg .jpeg .tif .tiff .bmp`.

---

## 4. Fundus-disc cropping

Fundus-disc cropping is **ON by default**. The preprocessor locates the circular
retinal region and crops it to a square at native resolution. Pass `--no_crop`
to disable cropping (e.g. when your images are already tightly cropped).

| Flag              | Behaviour                                       |
| ----------------- | ----------------------------------------------- |
| *(default)*       | fundus-disc crop → model resize                 |
| `--no_crop`       | model resize only                               |

The torchvision `transforms.Resize` step inside the dataloader is the sole
resize — the fundus-disc crop step keeps the image at its native cropped
resolution. This avoids an earlier double-resize (OpenCV `INTER_AREA` crop →
PIL `BILINEAR` resize) that could subtly diverge from training, where only
`transforms.Resize` was applied.

The training commands referenced in the project all use `--no_crop`; keep this
consistent at inference time to reproduce published AUCs.

---

## 5. Running inference

```bash
python infer_aviri.py \
    --backbone  <vit_base|swin_v2|retfound|retfound_mae|resnet50|vit_large_mae|vit_base_mae|dino_vitbase16|retizero|visionfm> \
    --method    <softmax|mc_dropout|edl> \
    --checkpoint <path/to/model.pth> \
    --input_dir  <path/to/images> \
    [--no_crop] [--batch_size 8] [--workers 4] [--mc_T 20] \
    [--device auto|cuda|cpu] [--output results.csv]
```

### 5.1 Softmax / predictive entropy

```bash
python infer_aviri.py --method softmax --backbone vit_base \
    --checkpoint models_weight/vit_base/aviri_class_model4vit_base_lr1e-05-AUC9428.pth \
    --no_crop --input_dir /path/to/images

python infer_aviri.py --method softmax --backbone retfound \
    --checkpoint models_weight/retfound/checkpoint-best_SEEDonly_AUC9497.pth \
    --no_crop --input_dir /path/to/images

# retfound_mae uses the same SEED-only checkpoint with drop_path=0.2 + BICUBIC resize
python infer_aviri.py --method softmax --backbone retfound_mae \
    --checkpoint models_weight/retfound/checkpoint-best_SEEDonly_AUC9497.pth \
    --no_crop --input_dir /path/to/images

python infer_aviri.py --method softmax --backbone swin_v2 \
    --checkpoint models_weight/swin_v2/aviri_class_model4swin_v2_lr1e-05-AUC9514.pth \
    --no_crop --input_dir /path/to/images

python infer_aviri.py --method softmax --backbone resnet50 \
    --checkpoint models_weight/resnet50/aviri_class_model4ResNet50_lr1e-05-AUC9454.pth \
    --no_crop --input_dir /path/to/images

# ViT-Large MAE (ImageNet-MAE pretrain)
python infer_aviri.py --method softmax --backbone vit_large_mae \
    --checkpoint models_weight/vit_large_mae/aviri_class_model4vit_large_mae_lr3e-05-AUC9505.pth \
    --no_crop --input_dir /path/to/images

# ViT-Base DINO (ImageNet-DINO pretrain)
python infer_aviri.py --method softmax --backbone dino_vitbase16 \
    --checkpoint models_weight/dino_vitbase16/aviri_class_model4dino_vitbase16_lr1e-06-AUC9472.pth \
    --no_crop --input_dir /path/to/images

# RetiZero (ViT-L/16 + LoRA rank 8, retinal-image CLIP pretrain)
python infer_aviri.py --method softmax --backbone retizero \
    --checkpoint models_weight/retizero/retizero_lr1e-4_best-AUC9443.pth \
    --no_crop --input_dir /path/to/images

# VisionFM (ViT-B/16 + 3-layer MLP over last-4-block CLS tokens,
# retinal-image DINO pretrain — uses VisionFM's Fundus mean/std + BICUBIC)
python infer_aviri.py --method softmax --backbone visionfm \
    --checkpoint models_weight/visionfm/checkpoint_best_finetune-AUC0.9484.pth \
    --no_crop --input_dir /path/to/images
```

### 5.2 MC Dropout (stochastic passes)

```bash
python infer_aviri.py --method mc_dropout --backbone vit_base \
    --checkpoint models_weight/vit_base/aviri_class_model4vit_base_mc_dropout_lr1e-05-AUC9437.pth \
    --mc_T 20 --no_crop --input_dir /path/to/images

python infer_aviri.py --method mc_dropout --backbone swin_v2 \
    --checkpoint models_weight/swin_v2/aviri_class_model4swin_v2_mc_dropout_lr1e-05-AUC9518.pth \
    --mc_T 20 --no_crop --input_dir /path/to/images

python infer_aviri.py --method mc_dropout --backbone retfound \
    --checkpoint models_weight/retfound/aviri_class_model4retfound_mc_dropout_lr5e-05-AUC9493.pth \
    --mc_T 20 --no_crop --input_dir /path/to/images
```

`--mc_T` controls how many stochastic forward passes are averaged (default 20,
matches training-time setting).

### 5.3 Evidential Deep Learning (EDL)

```bash
python infer_aviri.py --method edl --backbone vit_base \
    --checkpoint models_weight/vit_base/aviri_class_model4vit_base_edl_lr1e-05-AUC9464.pth \
    --no_crop --input_dir /path/to/images

python infer_aviri.py --method edl --backbone swin_v2 \
    --checkpoint models_weight/swin_v2/aviri_class_model4swin_v2_edl_lr1e-06-AUC9444.pth \
    --no_crop --input_dir /path/to/images

python infer_aviri.py --method edl --backbone retfound \
    --checkpoint models_weight/retfound/aviri_class_model4retfound_edl_lr5e-05-AUC9489.pth \
    --no_crop --input_dir /path/to/images
```

---

## 6. CPU vs GPU

By default (`--device auto`) the script uses CUDA if available, otherwise CPU.
Force a specific device with `--device cuda` or `--device cpu`. To pick a GPU
when multiple are present, set `CUDA_VISIBLE_DEVICES` before the command:

```bash
CUDA_VISIBLE_DEVICES=0 python infer_aviri.py ...   # GPU 0
```

CPU inference works for all three backbones (RETFound ViT-L is slow on CPU —
expect ~1–2 s/image).

---

## 7. Output

A CSV named `aviri_<backbone>_<method>.csv` (or whatever `--output` points to)
is written to the current directory with one row per image:

| Column                 | Meaning                                                       |
| ---------------------- | ------------------------------------------------------------- |
| `filename`             | Image file name                                               |
| `path`                 | Full path to the image                                        |
| `aviri_GT`             | Ground-truth label (0/1) or `N/A` when no `normal/`+`vi/`     |
| `aviri_prediction`     | Predicted probability of class 1 (visually impaired)          |
| `predictive_entropy`   | Predictive entropy of the output distribution (all methods)   |
| `expected_entropy`     | *(mc_dropout, edl)* Expected per-sample entropy               |
| `mutual_information`   | *(mc_dropout)* Epistemic uncertainty (pred − expected entropy)|
| `probability_variance` | *(mc_dropout)* Variance of class-1 probability across passes  |
| `edl_vacuity`          | *(edl)* `K / S` — low total evidence                          |
| `edl_expected_entropy` | *(edl)* Dirichlet expected entropy                            |
| `edl_mutual_information` | *(edl)* Dirichlet epistemic uncertainty                     |
| `edl_strength`         | *(edl)* Dirichlet strength `S = sum(alpha)`                   |

All uncertainty columns follow the "larger = more uncertain" convention.

When ground-truth is available, AUC-ROC, AUC-PR, sensitivity, specificity, and
confusion matrices (at threshold 0.5 and at the Youden-optimal threshold) are
also printed to the console.

---

## 8. Troubleshooting

- **`ModuleNotFoundError: No module named 'models_vit'`** — run `infer_aviri.py`
  from inside the `model_inference/` folder (so that both `models_vit.py` and
  `own_models_5.py` are on `sys.path`). The script already adds its own folder
  automatically — this error only appears if the file was moved.
- **`checkpoint contains no Dropout/DropPath layers`** — the checkpoint was
  trained without MC-dropout layers. Use the `_mc_dropout_` checkpoints under
  `models_weight/<backbone>/` for `--method mc_dropout`.
- **`WARNING: --device cuda requested but CUDA is not available`** — no GPU is
  visible; the script falls back to CPU automatically.
- **`no supported images found under ...`** — check the folder path and that
  files have one of the supported extensions.
