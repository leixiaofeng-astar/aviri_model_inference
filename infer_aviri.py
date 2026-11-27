"""AVIRI (Age-related Visual Impairment) — binary fundus classifier inference.

Supports four backbones (``vit_base``, ``swin_v2``, ``retfound``, ``resnet50``)
and three uncertainty-estimation methods (``softmax``, ``mc_dropout``, ``edl``).
Runs on GPU when available and falls back to CPU automatically.

Each input image is read as RGB, optionally fundus-disc cropped (cropping is ON
by default; pass ``--no_crop`` to disable), resized and normalised to the
backbone's input resolution, and then forwarded through the loaded model.

Folder layouts accepted via ``--input_dir``
-------------------------------------------
Flat (prediction only — no ground truth):

    input_dir/
        img001.jpg
        img002.png
        ...

Labelled (ground-truth evaluation):

    input_dir/
        normal/   <- label 0
        vi/       <- label 1

Examples
--------
# Softmax, Swin V2
python infer_aviri.py \
    --backbone swin_v2 \
    --method   softmax \
    --checkpoint models_weight/swin_v2/aviri_class_model4swin_v2_lr1e-05-AUC9514.pth \
    --input_dir /path/to/images

python infer_aviri.py \
    --backbone swin_v2 \
    --method   softmax \
    --checkpoint models_weight/swin_v2/aviri_class_model4swin_v2_lr1e-05-AUC9514.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test
python infer_aviri.py \
    --backbone swin_v2 \
    --method   softmax \
    --checkpoint models_weight/swin_v2/aviri_class_model4swin_v2_lr1e-05-AUC9514.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/seed_aviri/full_seed/test \
    --no_crop

python infer_aviri.py \
    --backbone swin_v2 \
    --method   softmax \
    --checkpoint models_weight/swin_v2/new_aviri_class_model4swin_v2_softmax_lr2e-06.pth \
    --input_dir /data/xiaofeng/HCSA/FM_compare/new_aviri_data/test --no_crop
python infer_aviri.py \
    --backbone swin_v2 \
    --method   softmax \
    --checkpoint models_weight/swin_v2/new_aviri_class_model4swin_v2_softmax_lr2e-06.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test

python infer_aviri.py \
    --backbone vit_base \
    --method   softmax \
    --checkpoint models_weight/vit_base/aviri_class_model4vit_base_lr1e-05-AUC9428.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test
python infer_aviri.py \
    --backbone vit_base \
    --method   softmax \
    --checkpoint models_weight/vit_base/aviri_class_model4vit_base_lr1e-05-AUC9428.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/seed_aviri/full_seed/test \
    --no_crop

python infer_aviri.py \
    --backbone vit_base \
    --method   softmax \
    --checkpoint models_weight/vit_base/new_aviri_class_model4vit_base_softmax_lr2e-06.pth \
    --input_dir /data/xiaofeng/HCSA/FM_compare/new_aviri_data/test --no_crop
python infer_aviri.py    \
    --backbone vit_base \
    --method   softmax \
    --checkpoint models_weight/vit_base/new_aviri_class_model4vit_base_softmax_lr2e-06.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test


python infer_aviri.py \
    --backbone retfound_mae \
    --method   softmax \
    --checkpoint models_weight/retfound/checkpoint-best_SEEDonly_AUC9497.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test
python infer_aviri.py \
    --backbone retfound_mae \
    --method   softmax \
    --checkpoint models_weight/retfound/checkpoint-best_SEEDonly_AUC9497.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/seed_aviri/full_seed/test \
    --no_crop

python infer_aviri.py \
    --backbone retfound \
    --method   softmax \
    --checkpoint models_weight/retfound/new_aviri_class_model4retfound_softmax_lr3e-05.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test
python infer_aviri.py \
    --backbone retfound \
    --method   softmax \
    --checkpoint models_weight/retfound/new_aviri_class_model4retfound_softmax_lr3e-05.pth \
    --input_dir /data/xiaofeng/HCSA/FM_compare/new_aviri_data/test \
    --no_crop

# Softmax, ResNet-50
python infer_aviri.py \
    --backbone resnet50 \
    --method   softmax \
    --checkpoint models_weight/resnet50/aviri_class_model4ResNet50_lr1e-05-AUC9454.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test
python infer_aviri.py \
    --backbone resnet50 \
    --method   softmax \
    --checkpoint models_weight/resnet50/aviri_class_model4ResNet50_lr1e-05-AUC9454.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/seed_aviri/full_seed/test \
    --no_crop
python infer_aviri.py \
    --backbone resnet50 \
    --method   softmax \
    --checkpoint models_weight/resnet50/aviri_class_model4ResNet50_lr1e-05-AUC9454.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/crystalvue-aviri/hcsa20/test-aug2026 \
    --no_crop

python infer_aviri.py \
    --backbone resnet50 \
    --method   softmax \
    --checkpoint models_weight/resnet50/aviri_class_model4ResNet50_lr1e-05-AUC9454.pth \
    --input_dir /data2/xiaofeng/Foundation_Model/RETFound_MAE/bes_aviri/test --no_crop


# MC dropout, RETFound
python infer_aviri.py \
    --backbone  retfound \
    --method    mc_dropout \
    --checkpoint models_weight/retfound/aviri_class_model4retfound_mc_dropout_lr5e-05-AUC9493.pth \
    --input_dir /path/to/images \
    --mc_T 20

# EDL, ViT-Base, no cropping
python infer_aviri.py \
    --backbone  vit_base \
    --method    edl \
    --checkpoint models_weight/vit_base/aviri_class_model4vit_base_edl_lr1e-05-AUC9464.pth \
    --input_dir /path/to/images \
    --no_crop
"""

import argparse
import contextlib
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fundus_preprocess import fundus_image_preprocess_class  # noqa: E402
from own_models_5 import get_model  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
MEAN_NORM = [0.485, 0.456, 0.406]
STD_NORM = [0.229, 0.224, 0.225]

BACKBONE_INPUT_SIZE = {
    'vit_base': 224,
    'retfound': 224,
    'retfound_mae': 224,
    'swin_v2': 256,
    'resnet50': 224,
}

# RETFound-MAE was trained with BICUBIC resize (MAE util/datasets.py build_transform);
# all other backbones here use torchvision's default BILINEAR.
_BICUBIC_BACKBONES = {'retfound_mae'}


def _resolve_device(flag):
    if flag == 'cpu':
        return torch.device('cpu')
    if flag == 'cuda':
        if not torch.cuda.is_available():
            print('WARNING: --device cuda requested but CUDA is not available; using CPU.')
            return torch.device('cpu')
        return torch.device('cuda')
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EvidentialClassifier(nn.Module):
    """Wraps a backbone to produce Dirichlet evidence (used by EDL inference).

    Mirrors the training-time definition: the backbone's K-class logits are
    pushed through softplus to obtain non-negative evidence e_k ≥ 0.
    """

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        return F.softplus(_logits_from_output(self.backbone(x)))


def _logits_from_output(output):
    if isinstance(output, (tuple, list)):
        output = output[-1]
    if isinstance(output, dict):
        output = output.get('logits', next(iter(output.values())))
    if output.ndim == 1:
        output = output[:, None]
    if output.shape[1] == 1:
        return torch.cat([-output, output], dim=1)
    return output


class FundusInferenceDataset(Dataset):
    """Dataset that applies optional fundus cropping and the model transforms."""

    def __init__(self, file_paths, labels, transform, preprocessor=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.preprocessor = preprocessor

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = int(self.labels[idx]) if self.labels[idx] is not None else -1
        if self.preprocessor is not None:
            try:
                arr = self.preprocessor.imread(path)
                with contextlib.redirect_stdout(io.StringIO()):
                    arr = self.preprocessor.image_crop(arr)
                image = Image.fromarray(arr.astype('uint8')).convert('RGB')
            except Exception as exc:
                print(f'Warning: preprocessing failed for {os.path.basename(path)}: {exc}')
                image = Image.open(path).convert('RGB')
        else:
            image = Image.open(path).convert('RGB')
        return self.transform(image), label, os.path.basename(path), path


def build_transform(image_size, backbone):
    interp = (transforms.InterpolationMode.BICUBIC
              if backbone in _BICUBIC_BACKBONES
              else transforms.InterpolationMode.BILINEAR)
    return transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=interp),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_NORM, STD_NORM),
    ])


def discover_images(input_dir):
    """Return (paths, labels, has_gt).

    If ``input_dir/normal`` and ``input_dir/vi`` both exist, use them to build a
    labelled dataset. Otherwise, treat ``input_dir`` as a flat folder and
    return ``has_gt=False``.
    """
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'input_dir not found: {input_dir}')

    normal_dir = root / 'normal'
    vi_dir = root / 'vi'
    if normal_dir.is_dir() and vi_dir.is_dir():
        paths, labels = [], []
        for folder, lbl in ((normal_dir, 0), (vi_dir, 1)):
            for name in sorted(os.listdir(folder)):
                if Path(name).suffix.lower() in IMAGE_EXTENSIONS:
                    paths.append(str(folder / name))
                    labels.append(lbl)
        return paths, labels, True

    paths = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            paths.append(str(p))
    return paths, [None] * len(paths), False


def load_standard_checkpoint(checkpoint_path, backbone, device):
    """Load a softmax or MC-dropout checkpoint onto ``device``.

    Handles two save formats:
      - ``torch.save(model, path)`` — full nn.Module pickle; used directly.
      - state_dict (dict with ``model_state`` / ``state_dict`` / ``model`` or
        bare parameters) — a fresh architecture is built and the weights are
        loaded, including RETFound's cls_token/pos_embed/blocks prefixing.
    """
    obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(obj, nn.Module):
        model = obj
    else:
        model = get_model(backbone, num_class=2)
        state = obj
        if isinstance(obj, dict):
            for k in ('model_state', 'state_dict', 'model'):
                if k in obj and isinstance(obj[k], dict):
                    state = obj[k]
                    break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        if backbone in ('retfound', 'retfound_mae') and any(
                k in ('cls_token', 'pos_embed') or k.startswith('blocks.') for k in state):
            state = {('backbone.' + k if not k.startswith('head.') else k): v
                     for k, v in state.items()}
        model.load_state_dict(state, strict=False)
    return model.to(device).eval()


def load_edl_checkpoint(checkpoint_path, backbone, device):
    """Load an EDL checkpoint, wrapping it in :class:`EvidentialClassifier`."""
    obj = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if isinstance(obj, EvidentialClassifier):
        model = obj
    elif isinstance(obj, nn.Module):
        model = EvidentialClassifier(obj)
    else:
        saved_backbone = obj.get('backbone', backbone) if isinstance(obj, dict) else backbone
        model = EvidentialClassifier(get_model(saved_backbone, num_class=2))
        state = obj
        if isinstance(obj, dict):
            for k in ('model_state', 'state_dict', 'model'):
                if k in obj and isinstance(obj[k], dict):
                    state = obj[k]
                    break
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def enable_mc_dropout(model):
    """Put Dropout and timm DropPath layers into train mode (BatchNorm stays eval)."""
    try:
        from timm.layers import DropPath
        dp_type = DropPath
    except ImportError:
        dp_type = None
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(); n += 1
        elif dp_type is not None and isinstance(m, dp_type):
            m.train(); n += 1
    if n == 0:
        raise RuntimeError(
            'checkpoint contains no Dropout/DropPath layers — MC dropout would be deterministic')
    print(f'MC dropout enabled on {n} stochastic layers')


def predictive_entropy(probs):
    p = probs.clamp_min(1e-12)
    return -(p * p.log()).sum(-1)


@torch.no_grad()
def infer_softmax(model, loader, device):
    probs_all, labels_all, names, paths = [], [], [], []
    for images, labels, batch_names, batch_paths in tqdm(loader, desc='softmax'):
        logits = _logits_from_output(model(images.to(device)))
        probs_all.append(F.softmax(logits, dim=1).cpu())
        labels_all.append(labels)
        names.extend(batch_names)
        paths.extend(batch_paths)
    probs = torch.cat(probs_all)
    labels = torch.cat(labels_all)
    unc = {'predictive_entropy': predictive_entropy(probs)}
    return probs, labels, names, paths, unc


@torch.no_grad()
def infer_mc_dropout(model, loader, device, passes):
    enable_mc_dropout(model)
    probs_all, labels_all, names, paths = [], [], [], []
    for images, labels, batch_names, batch_paths in tqdm(loader, desc='mc_dropout'):
        images = images.to(device)
        samples = torch.stack([
            F.softmax(_logits_from_output(model(images)), dim=1) for _ in range(passes)
        ])  # (T, B, C)
        probs_all.append(samples.cpu())
        labels_all.append(labels)
        names.extend(batch_names)
        paths.extend(batch_paths)
    samples = torch.cat(probs_all, dim=1)  # (T, N, C)
    mean_probs = samples.mean(0)
    pe = predictive_entropy(mean_probs)
    ee = predictive_entropy(samples).mean(0)
    unc = {
        'predictive_entropy': pe,
        'expected_entropy': ee,
        'mutual_information': pe - ee,
        'probability_variance': samples[:, :, 1].var(0, unbiased=False),
    }
    return mean_probs, torch.cat(labels_all), names, paths, unc


@torch.no_grad()
def infer_edl(model, loader, device):
    probs_all, labels_all, names, paths = [], [], [], []
    vacuity, pred_ent, exp_ent, strength = [], [], [], []
    for images, labels, batch_names, batch_paths in tqdm(loader, desc='edl'):
        alpha = model(images.to(device)) + 1.0
        total = alpha.sum(1, keepdim=True)
        probs = alpha / total
        pe = predictive_entropy(probs)
        de = (probs * (torch.digamma(total + 1.0) - torch.digamma(alpha + 1.0))).sum(1)
        probs_all.append(probs.cpu())
        vacuity.append((alpha.shape[1] / total.squeeze(1)).cpu())
        pred_ent.append(pe.cpu())
        exp_ent.append(de.cpu())
        strength.append(total.squeeze(1).cpu())
        labels_all.append(labels)
        names.extend(batch_names)
        paths.extend(batch_paths)
    unc = {
        'predictive_entropy': torch.cat(pred_ent),
        'edl_vacuity': torch.cat(vacuity),
        'edl_expected_entropy': torch.cat(exp_ent),
        'edl_mutual_information': torch.cat(pred_ent) - torch.cat(exp_ent),
        'edl_strength': torch.cat(strength),
    }
    return torch.cat(probs_all), torch.cat(labels_all), names, paths, unc


def report_metrics(labels, scores):
    from sklearn.metrics import (auc, average_precision_score, confusion_matrix,
                                 precision_recall_curve, roc_auc_score, roc_curve)
    y, s = labels.numpy(), scores.numpy()
    roc_val = float(roc_auc_score(y, s))
    pr_val = float(average_precision_score(y, s))
    fpr, tpr, thr = roc_curve(y, s, pos_label=1)
    pr_prec, pr_rec, _ = precision_recall_curve(y, s)
    pr_curve_auc = float(auc(pr_rec, pr_prec))
    youden_idx = int(np.argmax(tpr - fpr))
    youden_thr = float(thr[youden_idx]) if np.isfinite(thr[youden_idx]) else 0.5

    def cm_stats(t):
        pred = (s >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else 0.0
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        return int(tn), int(fp), int(fn), int(tp), sens, spec

    tn5, fp5, fn5, tp5, sens5, spec5 = cm_stats(0.5)
    tny, fpy, fny, tpy, sensy, specy = cm_stats(youden_thr)

    print()
    print('=' * 62)
    print(f'  AUC-ROC        : {roc_val:.4f}')
    print(f'  AUC-PR (AP)    : {pr_val:.4f}')
    print(f'  AUC-PR (curve) : {pr_curve_auc:.4f}')
    print()
    print(f'  Threshold = 0.5')
    print(f'    Sensitivity : {sens5:.4f}')
    print(f'    Specificity : {spec5:.4f}')
    print(f'    Confusion   : TN={tn5}  FP={fp5}  FN={fn5}  TP={tp5}')
    print()
    print(f'  Threshold = {youden_thr:.4f} (optimal Youden)')
    print(f'    Sensitivity : {sensy:.4f}')
    print(f'    Specificity : {specy:.4f}')
    print(f'    Confusion   : TN={tny}  FP={fpy}  FN={fny}  TP={tpy}')
    print('=' * 62)
    return {'roc_auc': roc_val, 'pr_auc': pr_val, 'youden_threshold': youden_thr}


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--backbone', required=True,
                        choices=['vit_base', 'swin_v2', 'retfound', 'retfound_mae', 'resnet50'],
                        help='Model backbone. retfound_mae = ViT-L/16 with drop_path=0.2 and '
                             'BICUBIC resize (matches RETFound-MAE main_finetune.py defaults). '
                             'resnet50 = torchvision ResNet-50 (224x224 input)')
    parser.add_argument('--method', default='softmax',
                        choices=['softmax', 'mc_dropout', 'edl'],
                        help='Inference / uncertainty method')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to the trained .pth checkpoint')
    parser.add_argument('--input_dir', required=True,
                        help='Image folder (flat for predictions only, or with normal/ and vi/ '
                             'subfolders for ground-truth evaluation)')
    parser.add_argument('--no_crop', action='store_true',
                        help='Disable fundus-disc crop preprocessing (crop is ON by default)')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--mc_T', type=int, default=20,
                        help='Number of stochastic passes for --method mc_dropout')
    parser.add_argument('--device', default='auto', choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--output', default=None,
                        help='Output CSV path (default: aviri_<backbone>_<method>.csv)')
    return parser.parse_args()


def main():
    args = parse_args()
    device = _resolve_device(args.device)
    print(f'Device       : {device}')

    if not os.path.isfile(args.checkpoint):
        sys.exit(f'Error: checkpoint not found: {args.checkpoint}')

    paths, labels, has_gt = discover_images(args.input_dir)
    if not paths:
        sys.exit(f'Error: no supported images found under {args.input_dir}')

    image_size = BACKBONE_INPUT_SIZE[args.backbone]
    use_crop = not args.no_crop
    # do_resize=False so the fundus-disc crop keeps native resolution and the
    # torchvision transforms.Resize (BILINEAR / BICUBIC) is the sole resize
    # step — otherwise an OpenCV INTER_AREA resize would run first and the
    # transforms.Resize would re-sample on top of it, diverging from training.
    preprocessor = (fundus_image_preprocess_class(resize=image_size, do_resize=False)
                    if use_crop else None)
    transform = build_transform(image_size, args.backbone)

    print(f'Backbone     : {args.backbone}  (input {image_size}x{image_size})')
    print(f'Method       : {args.method}')
    print(f'Crop         : {"ON" if use_crop else "OFF (--no_crop)"}')
    print(f'Input folder : {args.input_dir}')
    if has_gt:
        pos = sum(1 for l in labels if l == 1)
        neg = sum(1 for l in labels if l == 0)
        print(f'Images       : {neg} normal + {pos} vi = {len(paths)} total')
    else:
        print(f'Images       : {len(paths)} (prediction only — no ground truth)')

    dataset = FundusInferenceDataset(paths, labels, transform, preprocessor)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers)

    print(f'Checkpoint   : {args.checkpoint}')
    if args.method == 'edl':
        model = load_edl_checkpoint(args.checkpoint, args.backbone, device)
    else:
        model = load_standard_checkpoint(args.checkpoint, args.backbone, device)

    t0 = time.time()
    if args.method == 'softmax':
        probs, lbls, names, path_list, unc = infer_softmax(model, loader, device)
    elif args.method == 'mc_dropout':
        probs, lbls, names, path_list, unc = infer_mc_dropout(model, loader, device, args.mc_T)
    else:
        probs, lbls, names, path_list, unc = infer_edl(model, loader, device)
    elapsed = time.time() - t0
    n = len(probs)
    print(f'\nInference    : {n} images in {elapsed:.1f}s ({n / elapsed:.1f} img/s)')

    scores = probs[:, 1]
    frame = pd.DataFrame({
        'filename': names,
        'path': path_list,
        'aviri_GT': lbls.numpy() if has_gt else ['N/A'] * n,
        'aviri_prediction': scores.numpy(),
    })
    for key, value in unc.items():
        frame[key] = value.numpy()

    if has_gt:
        report_metrics(lbls, scores)

    output_csv = args.output or f'aviri_{args.backbone}_{args.method}.csv'
    frame.to_csv(output_csv, index=False)
    print(f'\nResults saved to: {output_csv}')


if __name__ == '__main__':
    main()
