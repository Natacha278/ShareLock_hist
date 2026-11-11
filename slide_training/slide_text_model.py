"""
Report-guided slide embedding + CLIP InfoNCE, in the spirit of LIMIT / CONCH-Z
(Locked Image Multiple Instance PreTraining: frozen patch space, report embeddings as cross-attention
queries over the bag of patch features, only the text side is trained).

image="locked" (default, LIMIT-like)
    The patch features are NOT transformed. The text MLP maps report features into the patch space
    (img_dim, like ShareLock project="text"), and the attention is parameter-free:
        t_i      = TextMLP(report_i)                                 [img_dim]
        a_ijn    = softmax_n( attn_scale * cos(t_i, p_jn) )          over the patches n of slide j
        s_ij     = sum_n a_ijn * p_jn                                slide j "read by" report i (raw patch space)
        logit_ij = exp(logit_scale) * cos(t_i, s_ij)
    Trainable: text MLP, attention temperature, logit scale. Because the patch space is untouched, the
    trained text MLP can be used directly against single patches (patch-level zero-shot), and it can be
    exported as a train_projection.py checkpoint (project="text").

image="learned"
    Patches go through a trainable MLP + LayerNorm and multi-head cross-attention with learned
    Wq/Wk/Wv/Wo in a `dim`-dimensional space (more capacity, but the text no longer lives in the patch space).

query="report" (default): the report is the attention query -> every slide of the batch is pooled with
    every report of the batch (B x B slide embeddings); loss = symmetric InfoNCE on logit_ij.
query="learned": learned query token(s) instead of the report (text-independent slide embedding) -> ablation.

    model = SlideTextModel(img_dim=1536, txt_dim=768)
    logits = model(patches, mask, txt)          # [B_txt, B_slide]
    loss = pairwise_clip_loss(logits)
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from projection import MLP


class SlideTextModel(nn.Module):
    def __init__(self, img_dim, txt_dim, image="locked", dim=512, heads=8, txt_depth=2, txt_width=1024,
                 txt_norm=True, txt_dropout=0.0, patch_depth=1, patch_width=1024, attn_dropout=0.0,
                 query="report", n_queries=1, init_attn_temperature=1.0, learn_attn_temperature=True,
                 init_temperature=0.07, learn_temperature=True, max_logit_scale=100.0):
        super().__init__()
        if image not in {"locked", "learned"}:
            raise ValueError(image)
        if query not in {"report", "learned"}:
            raise ValueError(query)
        self.image, self.query = image, query
        if image == "locked":
            dim, heads = img_dim, 1                       # everything lives in the frozen patch space
        elif dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim, self.heads = dim, heads
        self.txt_proj = MLP(txt_dim, dim, depth=txt_depth, width=txt_width, norm=txt_norm, dropout=txt_dropout)

        if image == "learned":
            self.patch_proj = MLP(img_dim, dim, depth=patch_depth, width=patch_width)
            self.norm_p, self.norm_q = nn.LayerNorm(dim), nn.LayerNorm(dim)
            self.wq, self.wk, self.wv, self.wo = (nn.Linear(dim, dim) for _ in range(4))
            self.attn_drop = nn.Dropout(attn_dropout)
        else:
            a = torch.tensor(math.log(1.0 / init_attn_temperature))
            if learn_attn_temperature:
                self.attn_log_scale = nn.Parameter(a)
            else:
                self.register_buffer("attn_log_scale", a)
        if query == "learned":
            self.queries = nn.Parameter(torch.randn(n_queries, dim) * dim ** -0.5)

        init = torch.tensor(math.log(1.0 / init_temperature))
        if learn_temperature:
            self.logit_scale = nn.Parameter(init)
        else:
            self.register_buffer("logit_scale", init)
        self.max_log_scale = math.log(max_logit_scale)

    # ------------------------------------------------------------------ pieces
    def scale(self):
        return self.logit_scale.clamp(max=self.max_log_scale).exp()

    @property
    def temperature(self):
        return 1.0 / self.scale().item()

    @property
    def attn_temperature(self):
        return 1.0 / self.attn_log_scale.clamp(max=self.max_log_scale).exp().item() if self.image == "locked" else None

    def encode_text(self, txt):
        """Report embedding before normalisation [T, dim] (also used as attention query)."""
        return self.txt_proj(txt)

    def keys_values(self, patches, mask):
        if not torch.is_autocast_enabled(patches.device.type):        # fp16 bags without autocast (e.g. CPU)
            patches = patches.to(self.txt_proj.net[0].weight.dtype)
        if self.image == "locked":
            return F.normalize(patches, dim=-1), patches                 # keys: unit patches, values: raw patches
        p = self.norm_p(self.patch_proj(patches))
        S, N, _ = p.shape
        return self.wk(p).view(S, N, self.heads, -1), self.wv(p).view(S, N, self.heads, -1)

    def pool(self, q_in, k, v, mask, return_attn=False):
        """q_in [T, dim] queries, mask [S, N] -> slide embeddings [T, S, dim] (+ attention [T, S, h, N])."""
        if self.image == "locked":
            q = F.normalize(q_in, dim=-1)
            s = self.attn_log_scale.clamp(max=self.max_log_scale).exp()
            scores = s * torch.einsum("td,snd->tsn", q, k.to(q.dtype))
            scores = scores.masked_fill(~mask[None], float("-inf"))
            attn = scores.float().softmax(-1).to(v.dtype)
            out = torch.einsum("tsn,snd->tsd", attn, v)
            return (out, attn[:, :, None]) if return_attn else out
        q = self.wq(self.norm_q(q_in)).view(q_in.shape[0], self.heads, -1)
        scores = torch.einsum("thd,snhd->tshn", q, k) / math.sqrt(q.shape[-1])
        scores = scores.masked_fill(~mask[None, :, None, :], float("-inf"))
        attn = self.attn_drop(scores.float().softmax(-1).to(v.dtype))
        out = torch.einsum("tshn,snhd->tshd", attn, v).reshape(q_in.shape[0], k.shape[0], self.dim)
        out = self.wo(out)
        return (out, attn) if return_attn else out

    def slide_embeddings(self, t, patches, mask):
        """Normalised slide embeddings [T, S, dim] (report query) or [1, S, dim] (learned query)."""
        k, v = self.keys_values(patches, mask)
        if self.query == "report":
            s = self.pool(t, k, v, mask)
        else:
            s = self.pool(self.queries, k, v, mask).mean(0, keepdim=True)   # average the learned queries
        return F.normalize(s.float(), dim=-1)

    # ------------------------------------------------------------------ forward
    def forward(self, patches, mask, txt):
        """patches [S, N, img_dim], mask [S, N], txt [T, txt_dim] -> logits [T, S]."""
        t = self.encode_text(txt)
        s = self.slide_embeddings(t, patches, mask)                      # [T or 1, S, dim]
        z_t = F.normalize(t.float(), dim=-1)
        return self.scale() * torch.einsum("td,tsd->ts", z_t, s.expand(len(t), -1, -1))

    # ------------------------------------------------------------------ interop with train_projection.py
    def projection_checkpoint(self, img_dim, txt_dim, txt_depth, txt_width, txt_norm, txt_dropout):
        """State dict + config loadable by projection.ProjectionModel(project="text") (locked image only)."""
        if self.image != "locked":
            raise ValueError("only image='locked' keeps the patch space, so only it maps to ProjectionModel")
        sd = {k: v.detach().cpu() for k, v in self.state_dict().items() if k.startswith("txt_proj.")}
        sd["logit_scale"] = self.logit_scale.detach().cpu()
        cfg = dict(img_dim=img_dim, txt_dim=txt_dim, project="text", embed_dim=None, depth=txt_depth,
                   width=txt_width, norm=txt_norm, dropout=txt_dropout,
                   init_temperature=self.temperature, learn_temperature=True)
        return sd, cfg


def pairwise_clip_loss(logits):
    """Symmetric InfoNCE on a [B, B] logit matrix whose diagonal holds the matching pairs."""
    target = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.t(), target))
