"""Model architecture definitions for the AVIRI visual-impairment inference pipeline.

This module must be importable at ``own_models_5`` (same module path used when
the checkpoints were pickled) so that ``torch.load`` can unpickle full-model
checkpoints saved via ``torch.save(model, path)``.

Supported backbones at inference time
-------------------------------------
  * ``vit_base``         → ViT-Base/16 ImageNet (224x224 input)
  * ``swin_v2``          → Swin V2 Base (256x256 input)
  * ``retfound``         → RETFound CFP ViT-L/16 (224x224 input)
  * ``retfound_mae``     → RETFound MAE CFP ViT-L/16, drop_path=0.2 (224x224 input)
  * ``resnet50``         → torchvision ResNet-50 (224x224 input)
  * ``vit_large_mae``    → ImageNet-MAE ViT-L/16 (224x224 input)
  * ``vit_base_mae``     → ImageNet-MAE ViT-B/16 (224x224 input)
  * ``dino_vitbase16``   → ImageNet-DINO ViT-B/16 (224x224 input)
  * ``retizero``         → RetiZero ViT-L/16 + LoRA, rank 8 (224x224 input)
  * ``visionfm``         → VisionFM ViT-B/16 + 3-layer MLP head over last-4-block CLS tokens (224x224 input)

None of the wrapper classes auto-download pretrained weights at inference
time — ``pretrained=False`` is passed to ``timm.create_model`` and the
fine-tuned task weights come from the user-supplied ``--checkpoint``.
"""

import os
import sys

import torch
import torch.nn as nn
import torchvision.models as tv_models
from timm.models import create_model
from timm.models.layers import trunc_normal_

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _inject_head_dropout(model, drop_rate):
    """Enable head dropout for MC Dropout inference.

    Mirrors the behaviour of the training-time helper so that MC Dropout
    checkpoints keep their stochastic classifier head.
    """
    if drop_rate <= 0.0:
        return
    if hasattr(model, 'head') and hasattr(model.head, 'drop'):
        model.head.drop = nn.Dropout(p=drop_rate)
    elif hasattr(model, 'head_drop'):
        model.head_drop = nn.Dropout(p=drop_rate)


class vit_base_window16_224(nn.Module):
    """ViT-Base/16 classifier — 224x224 input."""

    def __init__(self, num_class=2, drop_rate=0.0):
        super().__init__()
        self.model = create_model(
            'vit_base_patch16_224',
            pretrained=False,
            num_classes=num_class,
            global_pool='avg',
        )
        _inject_head_dropout(self.model, drop_rate)

    def forward(self, x):
        return self.model(x)


class swinv2_base_window12to16_256(nn.Module):
    """Swin Transformer V2 Base classifier — 256x256 input."""

    def __init__(self, num_class=2, drop_rate=0.0):
        super().__init__()
        self.model = create_model(
            'swinv2_base_window12to16_192to256',
            pretrained=False,
            num_classes=num_class,
            global_pool='avg',
        )
        _inject_head_dropout(self.model, drop_rate)

    def forward(self, x):
        return self.model(x)


class RETFound_cfp(nn.Module):
    """RETFound CFP ViT-L/16 classifier — 224x224 input.

    The pretraining weights are NOT fetched at inference time. For state-dict
    checkpoints the architecture is initialised and the fine-tuned weights are
    loaded by the inference pipeline. For full-model checkpoints Python's
    pickle protocol bypasses ``__init__`` and restores the module state
    directly, so this constructor is only invoked for state-dict builds.
    """

    def __init__(self, num_class=2, drop_rate=0.0, pretrained_path=None):
        super().__init__()
        from models_vit import vit_large_patch16

        self.backbone = vit_large_patch16(
            num_classes=0,
            drop_path_rate=0.1,
            global_pool=True,
        )
        self.head_drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else None
        self.head = nn.Linear(1024, num_class)
        trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)

        if pretrained_path and os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            cm = checkpoint.get('model', checkpoint)
            for suffix in ('weight', 'bias'):
                if f'norm.{suffix}' in cm:
                    cm[f'fc_norm.{suffix}'] = cm.pop(f'norm.{suffix}')
            encoder_state = {k: v for k, v in cm.items()
                             if not k.startswith('decoder') and not k.startswith('mask_token')}
            self.backbone.load_state_dict(encoder_state, strict=False)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if self.head_drop is not None:
            features = self.head_drop(features)
        return self.head(features)


class RETFound_mae_cfp(nn.Module):
    """RETFound MAE CFP ViT-L/16 classifier — 224x224 input, drop_path_rate=0.2.

    Shares the same ViT-L/16 architecture and weights as :class:`RETFound_cfp`
    but matches the ``main_finetune.py`` default of ``drop_path_rate=0.2`` and
    the BICUBIC-resize pipeline used by the newer RETFound-MAE repo. Pretraining
    weights are NOT fetched at inference time.
    """

    def __init__(self, num_class=2, drop_rate=0.0, pretrained_path=None):
        super().__init__()
        from models_vit import vit_large_patch16

        self.backbone = vit_large_patch16(
            num_classes=0,
            drop_path_rate=0.2,
            global_pool=True,
        )
        self.head_drop = nn.Dropout(p=drop_rate) if drop_rate > 0 else None
        self.head = nn.Linear(1024, num_class)
        trunc_normal_(self.head.weight, std=2e-5)
        nn.init.zeros_(self.head.bias)

        if pretrained_path and os.path.isfile(pretrained_path):
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            cm = checkpoint.get('model', checkpoint)
            for suffix in ('weight', 'bias'):
                if f'norm.{suffix}' in cm:
                    cm[f'fc_norm.{suffix}'] = cm.pop(f'norm.{suffix}')
            encoder_state = {k: v for k, v in cm.items()
                             if not k.startswith('decoder') and not k.startswith('mask_token')}
            self.backbone.load_state_dict(encoder_state, strict=False)

    def forward(self, x):
        features = self.backbone.forward_features(x)
        if self.head_drop is not None:
            features = self.head_drop(features)
        return self.head(features)


class vit_large_mae_window16_224(nn.Module):
    """ImageNet-1K MAE-pretrained ViT-L/16 classifier — 224x224 input.

    Matches ``FM_compare/own_models_5.py::vit_large_mae_window16_224`` so a
    state-dict build mirrors the training-time structure. ``pretrained=False``
    at inference time avoids any weight download — the fine-tuned ``.pth``
    supplies all tensors.
    """

    def __init__(self, num_class=2, drop_rate=0.0):
        super().__init__()
        self.model = create_model(
            'vit_large_patch16_224.mae',
            pretrained=False,
            num_classes=num_class,
            global_pool='avg',
        )
        _inject_head_dropout(self.model, drop_rate)

    def forward(self, x):
        return self.model(x)


# Alias used by the AVIRI vit_large_mae checkpoint (its pickle references
# ``own_models_5.vit_large_window16_224`` even though the pretrain source is
# MAE). Same architecture — just a different wrapper name at training time.
vit_large_window16_224 = vit_large_mae_window16_224


class vit_base_mae_window16_224(nn.Module):
    """ImageNet-1K MAE-pretrained ViT-B/16 classifier — 224x224 input."""

    def __init__(self, num_class=2, drop_rate=0.0):
        super().__init__()
        self.model = create_model(
            'vit_base_patch16_224.mae',
            pretrained=False,
            num_classes=num_class,
            global_pool='avg',
        )
        _inject_head_dropout(self.model, drop_rate)

    def forward(self, x):
        return self.model(x)


class vit_base_dino_window16_224(nn.Module):
    """ImageNet-1K DINO-pretrained ViT-B/16 classifier — 224x224 input."""

    def __init__(self, num_class=2, drop_rate=0.0):
        super().__init__()
        self.model = create_model(
            'vit_base_patch16_224.dino',
            pretrained=False,
            num_classes=num_class,
            global_pool='avg',
        )
        _inject_head_dropout(self.model, drop_rate)

    def forward(self, x):
        return self.model(x)


class ResNet50(nn.Module):
    """Torchvision ResNet-50 classifier — 224x224 input.

    Matches the module layout used when the AVIRI ResNet-50 checkpoint was
    pickled in ``FM_compare/own_models_5.py`` so ``torch.load`` can rebuild the
    full model (``self.features`` as the resnet backbone minus the classifier
    head, ``self.dropout``, and ``self.fc``). The forward returns raw
    ``self.fc`` logits — inference callers apply softmax themselves, so
    emitting probabilities here would cause a double-softmax.
    """

    def __init__(self, num_class=2, drop_rate=0.0):
        super().__init__()
        resnet = tv_models.resnet50(weights=None)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.dropout = nn.Dropout(p=drop_rate)
        self.fc = nn.Linear(in_features=resnet.fc.in_features, out_features=num_class)

    def forward(self, x):
        features = self.features(x).view(x.shape[0], -1)
        if hasattr(self, 'dropout'):
            features = self.dropout(features)
        return self.fc(features)


class _LoRA_qkv(nn.Module):
    """LoRA-adapter wrapper around a ViT block's fused ``attn.qkv`` linear.

    Minimal re-vendoring of ``RetiZero/iden_modules/modeling/LORA/lora_image_encoder.py``
    sufficient to rebuild the key layout saved in a fine-tuned RetiZero
    checkpoint: ``attn.qkv.qkv.weight/bias`` (original fused qkv) plus four
    low-rank adapters ``attn.qkv.linear_{a,b}_{q,v}.weight``.
    """

    def __init__(self, qkv, linear_a_q, linear_b_q, linear_a_v, linear_b_v):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :self.dim] = qkv[:, :, :self.dim] + new_q
        qkv[:, :, -self.dim:] = qkv[:, :, -self.dim:] + new_v
        return qkv


class _RetiZeroImgEncoder(nn.Module):
    """RetiZero image encoder: timm ViT-L/16 with every block's attn.qkv
    wrapped in :class:`_LoRA_qkv` (rank 8). Reproduces the
    ``img_encoder.lora_vit.*`` state-dict layout saved by
    ``RetiZero/finetune_retizero.py::ModelFinetuning``.
    """

    def __init__(self, r=8):
        super().__init__()
        vit = create_model(
            'vit_large_patch16_224',
            pretrained=False,
            num_classes=0,
            global_pool='token',
        )
        for blk in vit.blocks:
            w_qkv = blk.attn.qkv
            dim = w_qkv.in_features
            blk.attn.qkv = _LoRA_qkv(
                w_qkv,
                nn.Linear(dim, r, bias=False),
                nn.Linear(r, dim, bias=False),
                nn.Linear(dim, r, bias=False),
                nn.Linear(r, dim, bias=False),
            )
        self.lora_vit = vit

    def forward(self, x):
        return self.lora_vit(x)


class RetiZeroFinetune(nn.Module):
    """RetiZero (ViT-L/16 + LoRA, rank 8) classifier — 224x224 input.

    Matches the state-dict layout produced by
    ``RetiZero/finetune_retizero.py::ModelFinetuning`` so fine-tuned
    checkpoints load directly with ``strict=True``. No RetiZero pretrained
    CLIP weights are needed at inference — the fine-tuned ``.pth`` supplies
    every tensor (encoder + LoRA adapters + classifier).
    """

    def __init__(self, num_class=2, drop_rate=0.0, lora_r=8):
        super().__init__()
        self.img_encoder = _RetiZeroImgEncoder(r=lora_r)
        self.classifier = nn.Linear(1024, num_class, bias=True)

    def forward(self, x):
        feats = self.img_encoder(x)
        return self.classifier(feats)


def get_model(backbone, num_class=2, drop_rate=0.0):
    if backbone == 'vit_base':
        return vit_base_window16_224(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'swin_v2':
        return swinv2_base_window12to16_256(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'retfound':
        return RETFound_cfp(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'retfound_mae':
        return RETFound_mae_cfp(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'resnet50':
        return ResNet50(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'vit_large_mae':
        return vit_large_mae_window16_224(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'vit_base_mae':
        return vit_base_mae_window16_224(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'dino_vitbase16':
        return vit_base_dino_window16_224(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'retizero':
        return RetiZeroFinetune(num_class=num_class, drop_rate=drop_rate)
    if backbone == 'visionfm':
        from visionfm_model import VisionFMClassifier
        return VisionFMClassifier(num_class=num_class)
    raise ValueError(f"Unsupported backbone: {backbone}")
