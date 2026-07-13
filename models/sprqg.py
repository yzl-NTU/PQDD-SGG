
# PQDD-SGG : Semantic Prior Relation Query Generation (SPRQG)
import torch
from torch import nn

from .pqdd_utils import MLP, CrossAttentionBlock, build_token_reference_boxes


class SPRQG(nn.Module):
    """Semantic Prior Relation Query Generation module.

    Given the encoder token sequence ``F`` it:
      1. scores every token with an MLP classification head,
      2. keeps the ``num_rel_queries`` tokens with the highest objectness,
      3. duplicates them into subject / object queries,
      4. enhances them through bidirectional cross-attention,
      5. concatenates the two halves into relation queries ``Q_rel (M x 512)``.

    It additionally exposes the per-token classification logits and per-token
    boxes so the model can attach a DETR-style auxiliary loss (``L_aux``) that
    supervises the token-selection head.
    """

    def __init__(self, d_model=256, nhead=8, num_classes=151, num_rel_queries=200,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_rel_queries = num_rel_queries
        self.num_classes = num_classes

        # token scoring / selection head (the "classification head" with the
        # adaptive threshold tau in the paper).
        self.token_class_embed = MLP(d_model, d_model, num_classes + 1, 3)
        # a box head so the auxiliary loss can regress an entity box per token.
        self.token_bbox_embed = MLP(d_model, d_model, 4, 3)

        # bidirectional subject <-> object cross-attention
        self.sub2obj_attn = CrossAttentionBlock(d_model, nhead, dropout)
        self.obj2sub_attn = CrossAttentionBlock(d_model, nhead, dropout)

        # a small projection that lets the two query branches diverge from the
        # shared selected-token representation before cross-attention.
        self.sub_proj = nn.Linear(d_model, d_model)
        self.obj_proj = nn.Linear(d_model, d_model)

    def forward(self, memory, mask, spatial_shape):
        """
        Args:
            memory:        encoder tokens, ``(L, bs, d_model)`` with ``L = H*W``.
            mask:          padding mask, ``(bs, L)`` (True = padded).
            spatial_shape: ``(H, W)`` of the feature map.
        Returns:
            rel_queries:   ``(M, bs, 2*d_model)`` relation queries.
            sub_ref / obj_ref: ``(M, bs, 4)`` reference boxes for sub / obj queries.
            aux_outputs:   dict with per-token ``token_logits`` / ``token_boxes``
                           and the selected-token indices (for ``L_aux``).
        """
        L, bs, c = memory.shape
        device = memory.device
        M = self.num_rel_queries

        # ---- per-token scoring (objectness) --------------------------------
        token_logits = self.token_class_embed(memory)           # (L, bs, C+1)
        token_ref = build_token_reference_boxes(spatial_shape, device)  # (L, 4)
        token_ref = token_ref.unsqueeze(1).repeat(1, bs, 1)     # (L, bs, 4)
        token_boxes = (self.token_bbox_embed(memory) + _inverse_sigmoid(token_ref)).sigmoid()

        # objectness score = max foreground class probability.  Padded tokens
        # are masked out so they can never be selected.
        scores = token_logits[..., :-1].sigmoid().max(-1)[0]    # (L, bs)
        if mask is not None:
            scores = scores.masked_fill(mask.transpose(0, 1), -1.0)

        # ---- adaptive top-M token selection --------------------------------
        # In normal SGG training L = H*W >> M, but guard against tiny feature
        # maps (e.g. very small / heavily padded inputs) where L < M by tiling
        # the selected indices up to exactly M so downstream shapes stay fixed.
        k = min(M, L)
        topk_idx = torch.topk(scores, k, dim=0)[1]             # (k, bs)
        if k < M:
            reps = (M + k - 1) // k
            topk_idx = topk_idx.repeat(reps, 1)[:M]            # (M, bs)
        gather_idx = topk_idx.unsqueeze(-1).expand(-1, -1, c)   # (M, bs, c)
        selected = torch.gather(memory, 0, gather_idx)          # (M, bs, c)
        sel_ref = torch.gather(token_boxes, 0, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

        # ---- duplicate into subject / object queries -----------------------
        q_sub = self.sub_proj(selected)                         # (M, bs, c)
        q_obj = self.obj_proj(selected)                         # (M, bs, c)

        # ---- bidirectional cross-attention enhancement ---------------------
        # subject attends to object as context, and vice-versa (Eq. 7-10).
        q_sub_enh, _ = self.sub2obj_attn(q_sub, q_obj, q_obj)   # Q_sub + CA(Q_sub, Q_obj)
        q_obj_enh, _ = self.obj2sub_attn(q_obj, q_sub, q_sub)   # Q_obj + CA(Q_obj, Q_sub)

        # ---- concatenate into relation queries (Eq. 11) --------------------
        rel_queries = torch.cat([q_sub_enh, q_obj_enh], dim=-1)  # (M, bs, 2c)

        aux_outputs = {
            "token_logits": token_logits,      # (L, bs, C+1)
            "token_boxes": token_boxes,        # (L, bs, 4)
            "topk_idx": topk_idx,              # (M, bs)
        }
        return rel_queries, sel_ref, sel_ref.clone(), aux_outputs


def _inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
