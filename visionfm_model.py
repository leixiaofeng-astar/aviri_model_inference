"""Minimal vendored VisionFM ViT-Base + classifier for inference.

Reproduces the architecture used by
``VisionFM-main/inference_visionfm_for_multiclass_classification.py`` well
enough to load a fine-tuned checkpoint (``checkpoint_best_finetune*.pth``)
and run classification. Only the inference-time code paths are kept — no
masked image modeling, no flash attention, no DDP, no patch sampling.

Checkpoint layout the loader expects
------------------------------------
Top-level dict with keys ``visionfm_state_dict`` (ViT backbone — standard
timm-style key names: ``cls_token``, ``pos_embed``, ``patch_embed.proj.*``,
``blocks.N.attn.{qkv,proj}.*``, ``blocks.N.mlp.fc{1,2}.*``,
``blocks.N.norm{1,2}.*``, ``norm.*``) and ``classifier_state_dict`` (3-layer
MLP head wrapped in DDP — keys are prefixed with ``module.``).

Forward
-------
The classifier uses the CLS tokens from the last 4 transformer blocks
(VisionFM's ``--n_last_blocks=4``, ``--avgpool_patchtokens=0`` default),
concatenated into a 3072-dim vector before the ``ClsHead``.
"""
from functools import partial

import torch
import torch.nn as nn


class _PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class _Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.norm1 = norm_layer(dim)
        self.attn = _Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                               attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = _Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class VisionFMViTBase(nn.Module):
    """ViT-Base/16 inference backbone matching VisionFM's state_dict layout.

    Only the ``get_intermediate_layers`` path is used by the fine-tuned
    classifier head, so a fused ``forward`` is unnecessary here.
    """

    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(img_size=img_size, patch_size=patch_size,
                                       in_chans=3, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.)
        self.blocks = nn.ModuleList([
            _Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def _prepare_tokens(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        return self.pos_drop(x)

    def get_intermediate_layers(self, x, n=1):
        x = self._prepare_tokens(x)
        out = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                out.append(self.norm(x))
        return out


class VisionFMClsHead(nn.Module):
    """Three-layer MLP classifier head over a BatchNorm2d-normalised feature.

    Mirrors ``VisionFM-main/models/head.py::ClsHead(layers=3)``: the
    ``channel_bn`` is a ``BatchNorm2d`` applied on a 4D view ``(B, C, 1, 1)``
    of the concatenated last-n-blocks CLS tokens.
    """

    def __init__(self, embed_dim, num_classes, layers=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        if layers != 3:
            raise NotImplementedError('only layers=3 is supported for inference')
        c0, c1, c2, c3 = embed_dim, embed_dim // 2, embed_dim // 4, num_classes
        self.classifier = nn.Sequential(
            nn.Linear(c0, c1),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(c1, c2),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(c2, c3),
        )
        self.channel_bn = nn.BatchNorm2d(embed_dim, eps=1e-6, momentum=0.99)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2).unsqueeze(3)
        x = self.channel_bn(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class VisionFMClassifier(nn.Module):
    """End-to-end VisionFM fine-tuned classifier.

    Reproduces the training-time inference path from
    ``inference_visionfm_for_multiclass_classification.py``:
    ``get_intermediate_layers(x, n=4) → [cls_tokens] → concat → ClsHead``.

    ``load_finetuned`` is the counterpart of that script's
    ``utils.load_pretrained_weights`` + ``linear_classifier.load_state_dict``:
    it reads both ``visionfm_state_dict`` (backbone) and
    ``classifier_state_dict`` (head, stripping the DDP ``module.`` prefix)
    from a single fine-tuned ``.pth``.
    """

    def __init__(self, num_class=2, n_last_blocks=4):
        super().__init__()
        self.n_last_blocks = n_last_blocks
        self.backbone = VisionFMViTBase()
        self.head = VisionFMClsHead(embed_dim=self.backbone.embed_dim * n_last_blocks,
                                    num_classes=num_class, layers=3)

    def forward(self, x):
        feats = self.backbone.get_intermediate_layers(x, n=self.n_last_blocks)
        cls = [f[:, 0] for f in feats]
        concat = torch.cat(cls, dim=-1)
        return self.head(concat)

    def load_finetuned(self, checkpoint_path, map_location='cpu'):
        ckpt = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if 'visionfm_state_dict' not in ckpt:
            raise KeyError(
                f"'visionfm_state_dict' not found in {checkpoint_path} — "
                f"got keys: {sorted(ckpt.keys())[:10]}"
            )
        self.backbone.load_state_dict(ckpt['visionfm_state_dict'], strict=True)
        head_sd = ckpt.get('classifier_state_dict')
        if head_sd is None:
            raise KeyError(f"'classifier_state_dict' not found in {checkpoint_path}")
        head_sd = {k.removeprefix('module.'): v for k, v in head_sd.items()}
        self.head.load_state_dict(head_sd, strict=True)
