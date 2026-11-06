"""
Projection heads for frozen image / text features, with a learnable temperature.

    from projection import ProjectionModel, clip_loss
    model = ProjectionModel(img_dim=1536, txt_dim=768, project="text", depth=4, width=2048).cuda()
    z_img, z_txt, scale = model(img, txt)          # L2-normalised embeddings + exp(logit_scale)
    loss = clip_loss(z_img, z_txt, scale)

project:
    "both"  : image -> MLP -> embed_dim, text -> MLP -> embed_dim  (embed_dim required)
    "text"  : text -> MLP -> img_dim, image is only L2-normalised   (ShareLock-style)
    "image" : image -> MLP -> txt_dim, text is only L2-normalised
depth: number of Linear layers in each projected branch.
    depth=1 is a single linear map; depth>=2 is Linear -> [LayerNorm] -> act -> dropout, repeated,
    with hidden size `width`, and a final Linear to the output size.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, depth=2, width=1024, act=nn.GELU, norm=True, dropout=0.0):
        super().__init__()
        if depth < 1:
            raise ValueError("depth must be >= 1 (use project='image'/'text' to skip a branch)")
        layers, d = [], in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width)]
            if norm:
                layers += [nn.LayerNorm(width)]
            layers += [act()]
            if dropout > 0:
                layers += [nn.Dropout(dropout)]
            d = width
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ProjectionModel(nn.Module):
    def __init__(self, img_dim, txt_dim, project="both", embed_dim=None, depth=2, width=1024,
                 act=nn.GELU, norm=True, dropout=0.0,
                 init_temperature=0.07, learn_temperature=True, max_logit_scale=100.0):
        super().__init__()
        if project not in {"both", "image", "text"}:
            raise ValueError(f"project must be 'both', 'image' or 'text', got {project!r}")
        if project == "both" and embed_dim is None:
            raise ValueError("embed_dim is required when project='both'")
        if project == "text":
            embed_dim = img_dim           # text is mapped into the (frozen) image space
        elif project == "image":
            embed_dim = txt_dim           # image is mapped into the (frozen) text space

        mlp = dict(depth=depth, width=width, act=act, norm=norm, dropout=dropout)
        self.img_proj = MLP(img_dim, embed_dim, **mlp) if project in {"both", "image"} else nn.Identity()
        self.txt_proj = MLP(txt_dim, embed_dim, **mlp) if project in {"both", "text"} else nn.Identity()
        self.project, self.embed_dim = project, embed_dim

        # CLIP-style temperature: logits = exp(logit_scale) * cos_sim, with exp(logit_scale) = 1 / T
        init = torch.tensor(math.log(1.0 / init_temperature))
        if learn_temperature:
            self.logit_scale = nn.Parameter(init)
        else:
            self.register_buffer("logit_scale", init)
        self.max_log_scale = math.log(max_logit_scale)

    def encode_image(self, x):
        return F.normalize(self.img_proj(x), dim=-1)

    def encode_text(self, x):
        return F.normalize(self.txt_proj(x), dim=-1)

    def scale(self):
        return self.logit_scale.clamp(max=self.max_log_scale).exp()

    @property
    def temperature(self):
        return 1.0 / self.scale().item()

    def forward(self, img, txt):
        return self.encode_image(img), self.encode_text(txt), self.scale()


def clip_loss(z_img, z_txt, scale):
    """Symmetric InfoNCE; row i of z_img matches row i of z_txt."""
    logits = scale * z_img @ z_txt.t()
    target = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))


@torch.no_grad()
def zero_shot_logits(model, img_feats, class_txt_feats):
    """img_feats [N, D_img], class_txt_feats [C, D_txt] (e.g. averaged prompt features per class)."""
    return model.encode_image(img_feats) @ model.encode_text(class_txt_feats).t()