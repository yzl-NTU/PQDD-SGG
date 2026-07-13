
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def _get_clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)."""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CrossAttentionBlock(nn.Module):
    """A residual multi-head cross-attention block: y = LN(x + CA(x, mem)).

    Used both inside SPRQG (bidirectional subject<->object attention) and as
    the basic visual cross-attention used in the decoder.  We use a vanilla
    ``nn.MultiheadAttention`` (instead of a deformable attention) so the code
    runs without compiling any custom CUDA op.  ``batch_first`` is False, i.e.
    tensors are ``(length, batch, dim)``.
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, query, key, value,
                query_pos: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None):
        q = self.with_pos_embed(query, query_pos)
        k = self.with_pos_embed(key, key_pos)
        attn_out, attn_weight = self.attn(q, k, value=value,
                                          key_padding_mask=key_padding_mask)
        out = query + self.dropout(attn_out)
        out = self.norm(out)
        return out, attn_weight


class FFNBlock(nn.Module):
    """Residual feed-forward block: y = LN(x + FFN(x))."""

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm(x)
        return x


class SelfAttentionBlock(nn.Module):
    """Residual multi-head self-attention block: y = LN(x + SA(x))."""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, x, pos: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        q = k = self.with_pos_embed(x, pos)
        x2 = self.attn(q, k, value=x, attn_mask=attn_mask)[0]
        x = x + self.dropout(x2)
        x = self.norm(x)
        return x


@torch.no_grad()
def build_token_reference_boxes(spatial_shape, device):
    """Return a normalised (cx, cy, w, h) anchor box for every encoder token.

    Each token in an ``H x W`` feature map is assigned an anchor centred on its
    grid cell with a default size of (1/W, 1/H).  These anchors give SPRQG a
    coarse spatial prior that the auxiliary encoder loss can regress against.
    """
    h, w = spatial_shape
    yy, xx = torch.meshgrid(
        torch.linspace(0.5, h - 0.5, h, device=device),
        torch.linspace(0.5, w - 0.5, w, device=device),
        indexing="ij",
    )
    cx = (xx / w).reshape(-1)
    cy = (yy / h).reshape(-1)
    bw = torch.full_like(cx, 1.0 / w)
    bh = torch.full_like(cy, 1.0 / h)
    return torch.stack([cx, cy, bw, bh], dim=-1)  # (H*W, 4)
