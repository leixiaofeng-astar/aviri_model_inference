"""Model architecture definitions for the AVIRI visual-impairment inference pipeline.

This module must be importable at ``own_models_5`` (same module path used when
the checkpoints were pickled) so that ``torch.load`` can unpickle full-model
checkpoints saved via ``torch.save(model, path)``.

Supported backbones at inference time
-------------------------------------
  * ``vit_base``  → ViT-Base/16 (224x224 input)
  * ``swin_v2``   → Swin V2 Base (256x256 input)
  * ``retfound``  → RETFound CFP ViT-L/16 (224x224 input)
  * ``resnet50``  → torchvision ResNet-50 (224x224 input)

RETFound_cfp does NOT auto-download pretraining weights at inference; the
trained task weights come from the user-supplied ``--checkpoint``.
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
    raise ValueError(f"Unsupported backbone: {backbone}")
