# ------------------------------------------------------------------------
# PQDD-SGG : Prior Guided Relation Queries with Decoupled Decoding
#            for One-Stage Scene Graph Generation
# ------------------------------------------------------------------------
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list, accuracy,
                       get_world_size, is_dist_avail_and_initialized)

from .backbone import build_backbone
from .encoder import TransformerEncoder
from .sprqg import SPRQG
from .dcd import DCD
from .matcher_pqdd import build_matchers


class PQDDSGG(nn.Module):
    """Prior Guided Relation Queries with Decoupled Decoding for SGG."""

    def __init__(self, backbone, num_classes, num_rel_classes, num_rel_queries=200,
                 hidden_dim=256, nheads=8, enc_layers=6, dim_feedforward=2048,
                 dropout=0.1, t1=4, t2=4, use_sgq=True):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_rel_queries = num_rel_queries

        self.backbone = backbone
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        self.encoder = TransformerEncoder(
            d_model=hidden_dim, nhead=nheads, num_layers=enc_layers,
            dim_feedforward=dim_feedforward, dropout=dropout)

        self.sprqg = SPRQG(d_model=hidden_dim, nhead=nheads, num_classes=num_classes,
                           num_rel_queries=num_rel_queries, dropout=dropout)

        self.dcd = DCD(d_model=hidden_dim, nhead=nheads, dim_feedforward=dim_feedforward,
                       dropout=dropout, num_rel_queries=num_rel_queries,
                       num_classes=num_classes, num_rel_classes=num_rel_classes,
                       t1=t1, t2=t2, use_sgq=use_sgq)

    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, pos = self.backbone(samples)

        src, mask = features[-1].decompose()
        assert mask is not None
        src = self.input_proj(src)                       # (bs, C, H, W)
        bs, c, h, w = src.shape

        src_flat = src.flatten(2).permute(2, 0, 1)       # (HW, bs, C)
        pos_flat = pos[-1].flatten(2).permute(2, 0, 1)   # (HW, bs, C)
        mask_flat = mask.flatten(1)                      # (bs, HW)

        # ---- encoder (global context) -------------------------------------
        memory = self.encoder(src_flat, src_key_padding_mask=mask_flat, pos=pos_flat)

        # ---- SPRQG : build prior-rich relation queries --------------------
        rel_queries, _, _, enc_aux = self.sprqg(memory, mask_flat, (h, w))

        # ---- DCD : coarse-to-fine cascade decoding ------------------------
        rclm_outputs, rrdm_outputs = self.dcd(
            rel_queries, memory, memory_pos=pos_flat,
            memory_key_padding_mask=mask_flat)

        final = rrdm_outputs[-1]

        # entity-detection outputs (subject + object union) for the COCO metric
        pred_logits = torch.cat([final["sub_logits"], final["obj_logits"]], dim=1)
        pred_boxes = torch.cat([final["sub_boxes"], final["obj_boxes"]], dim=1)

        out = {
            "sub_logits": final["sub_logits"], "sub_boxes": final["sub_boxes"],
            "obj_logits": final["obj_logits"], "obj_boxes": final["obj_boxes"],
            "rel_logits": final["rel_logits"],
            "pred_logits": pred_logits, "pred_boxes": pred_boxes,
            "rclm_outputs": rclm_outputs,
            "rrdm_outputs": rrdm_outputs,
            "enc_outputs": {
                "token_logits": enc_aux["token_logits"].transpose(0, 1),  # (bs, L, C+1)
                "token_boxes": enc_aux["token_boxes"].transpose(0, 1),    # (bs, L, 4)
            },
        }
        return out


# =====================================================================
#  Criterion : one-to-many + one-to-one + auxiliary (SPRQG) losses
# =====================================================================
class SetCriterionPQDD(nn.Module):
    """Decoupled cascade loss for PQDD-SGG.

    The one-to-many losses supervise every RCLM layer (coarse screening) and the
    one-to-one losses supervise every RRDM layer (refinement).  The stop-gradient
    inside the model guarantees the two stages receive non-conflicting gradients
    (no ROT).  An auxiliary entity loss supervises the SPRQG token-selection head.
    """

    def __init__(self, num_classes, num_rel_classes, matchers, weight_dict, eos_coef):
        super().__init__()
        self.num_classes = num_classes
        self.num_rel_classes = num_rel_classes
        self.o2m_matcher, self.o2o_matcher, self.enc_matcher = matchers
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef

        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

        empty_weight_rel = torch.ones(num_rel_classes + 1)
        empty_weight_rel[-1] = eos_coef
        self.register_buffer("empty_weight_rel", empty_weight_rel)

    # ---- helpers -------------------------------------------------------
    @staticmethod
    def _src_idx(indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _stage_losses(self, out, targets, indices, prefix, log=False):
        """Compute cls / rel / bbox / giou losses for a single cascade stage."""
        device = out["sub_logits"].device
        sub_logits = out["sub_logits"]
        obj_logits = out["obj_logits"]
        rel_logits = out["rel_logits"]

        # move matched indices to device
        idx = self._src_idx(indices)
        batch_idx = idx[0].to(device)
        src_idx = idx[1].to(device)

        # ---- target class tensors -------------------------------------
        tgt_sub = torch.full(sub_logits.shape[:2], self.num_classes,
                             dtype=torch.int64, device=device)
        tgt_obj = torch.full(obj_logits.shape[:2], self.num_classes,
                             dtype=torch.int64, device=device)
        tgt_rel = torch.full(rel_logits.shape[:2], self.num_rel_classes,
                             dtype=torch.int64, device=device)

        if src_idx.numel() > 0:
            sub_cls_o = torch.cat([t["labels"][t["rel_annotations"][J, 0]]
                                   for t, (_, J) in zip(targets, indices)]).to(device)
            obj_cls_o = torch.cat([t["labels"][t["rel_annotations"][J, 1]]
                                   for t, (_, J) in zip(targets, indices)]).to(device)
            rel_cls_o = torch.cat([t["rel_annotations"][J, 2]
                                   for t, (_, J) in zip(targets, indices)]).to(device)
            tgt_sub[(batch_idx, src_idx)] = sub_cls_o
            tgt_obj[(batch_idx, src_idx)] = obj_cls_o
            tgt_rel[(batch_idx, src_idx)] = rel_cls_o

        loss_cls = (F.cross_entropy(sub_logits.transpose(1, 2), tgt_sub, self.empty_weight)
                    + F.cross_entropy(obj_logits.transpose(1, 2), tgt_obj, self.empty_weight))
        loss_rel = F.cross_entropy(rel_logits.transpose(1, 2), tgt_rel, self.empty_weight_rel)

        losses = {f"loss_cls_{prefix}": loss_cls, f"loss_rel_{prefix}": loss_rel}

        # ---- box / giou losses (matched only) -------------------------
        num = max(src_idx.numel(), 1)
        if src_idx.numel() > 0:
            sub_box = out["sub_boxes"][(batch_idx, src_idx)]
            obj_box = out["obj_boxes"][(batch_idx, src_idx)]
            tgt_sub_box = torch.cat([t["boxes"][t["rel_annotations"][J, 0]]
                                     for t, (_, J) in zip(targets, indices)]).to(device)
            tgt_obj_box = torch.cat([t["boxes"][t["rel_annotations"][J, 1]]
                                     for t, (_, J) in zip(targets, indices)]).to(device)

            loss_bbox = (F.l1_loss(sub_box, tgt_sub_box, reduction="sum")
                         + F.l1_loss(obj_box, tgt_obj_box, reduction="sum")) / num
            loss_giou = (
                (1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(sub_box), box_ops.box_cxcywh_to_xyxy(tgt_sub_box)))).sum()
                + (1 - torch.diag(box_ops.generalized_box_iou(
                    box_ops.box_cxcywh_to_xyxy(obj_box), box_ops.box_cxcywh_to_xyxy(tgt_obj_box)))).sum()
            ) / num
        else:
            loss_bbox = sub_logits.sum() * 0.0
            loss_giou = sub_logits.sum() * 0.0

        losses[f"loss_bbox_{prefix}"] = loss_bbox
        losses[f"loss_giou_{prefix}"] = loss_giou

        if log and src_idx.numel() > 0:
            losses["sub_error"] = 100 - accuracy(sub_logits[(batch_idx, src_idx)], sub_cls_o)[0]
            losses["obj_error"] = 100 - accuracy(obj_logits[(batch_idx, src_idx)], obj_cls_o)[0]
            losses["rel_error"] = 100 - accuracy(rel_logits[(batch_idx, src_idx)], rel_cls_o)[0]
            losses["class_error"] = (losses["sub_error"] + losses["obj_error"]) / 2
        return losses

    def _avg_over_layers(self, stage_outputs, targets, matcher, prefix, log_last=False):
        """Apply a matcher + losses to every cascade layer and average."""
        acc = None
        n = len(stage_outputs)
        for li, out in enumerate(stage_outputs):
            indices = matcher(out, targets)
            losses = self._stage_losses(out, targets, indices, prefix,
                                        log=(log_last and li == n - 1))
            if acc is None:
                acc = {k: v for k, v in losses.items()}
            else:
                for k, v in losses.items():
                    if k.startswith("loss_"):
                        acc[k] = acc[k] + v
                    else:               # error logs: keep the last layer only
                        acc[k] = v
        # average the loss terms across layers
        for k in list(acc.keys()):
            if k.startswith("loss_"):
                acc[k] = acc[k] / n
        return acc

    def _enc_aux_loss(self, enc_out, targets):
        """Auxiliary entity loss that supervises the SPRQG token-selection head."""
        token_logits = enc_out["token_logits"]   # (bs, L, C+1)
        token_boxes = enc_out["token_boxes"]      # (bs, L, 4)
        device = token_logits.device
        indices = self.enc_matcher(token_logits, token_boxes, targets)
        idx = self._src_idx(indices)
        batch_idx, src_idx = idx[0].to(device), idx[1].to(device)

        tgt_cls = torch.full(token_logits.shape[:2], self.num_classes,
                             dtype=torch.int64, device=device)
        if src_idx.numel() > 0:
            cls_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).to(device)
            tgt_cls[(batch_idx, src_idx)] = cls_o
        loss_cls = F.cross_entropy(token_logits.transpose(1, 2), tgt_cls, self.empty_weight)

        num = max(src_idx.numel(), 1)
        if src_idx.numel() > 0:
            box = token_boxes[(batch_idx, src_idx)]
            tgt_box = torch.cat([t["boxes"][J] for t, (_, J) in zip(targets, indices)]).to(device)
            loss_bbox = F.l1_loss(box, tgt_box, reduction="sum") / num
            loss_giou = (1 - torch.diag(box_ops.generalized_box_iou(
                box_ops.box_cxcywh_to_xyxy(box), box_ops.box_cxcywh_to_xyxy(tgt_box)))).sum() / num
        else:
            loss_bbox = token_logits.sum() * 0.0
            loss_giou = token_logits.sum() * 0.0
        return {"loss_cls_enc": loss_cls, "loss_bbox_enc": loss_bbox, "loss_giou_enc": loss_giou}

    @torch.no_grad()
    def _cardinality(self, out, targets):
        rel_logits = out["rel_logits"]
        tgt_len = torch.as_tensor([len(t["rel_annotations"]) for t in targets],
                                  device=rel_logits.device)
        card_pred = (rel_logits.argmax(-1) != rel_logits.shape[-1] - 1).sum(1)
        return {"cardinality_error": F.l1_loss(card_pred.float(), tgt_len.float())}

    def forward(self, outputs, targets):
        losses = {}
        # one-to-many (RCLM) -- all T1 layers
        losses.update(self._avg_over_layers(
            outputs["rclm_outputs"], targets, self.o2m_matcher, "o2m", log_last=False))
        # one-to-one (RRDM) -- all T2 layers, log errors from the final layer
        losses.update(self._avg_over_layers(
            outputs["rrdm_outputs"], targets, self.o2o_matcher, "o2o", log_last=True))
        # auxiliary SPRQG / encoder loss
        losses.update(self._enc_aux_loss(outputs["enc_outputs"], targets))
        # logging only
        losses.update(self._cardinality(outputs["rrdm_outputs"][-1], targets))
        return losses


# =====================================================================
#  Post-processing (same as RelTR/DETR, used for the COCO bbox metric)
# =====================================================================
class PostProcess(nn.Module):
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        out_logits, out_bbox = outputs["pred_logits"], outputs["pred_boxes"]
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        return [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]


# =====================================================================
#  build
# =====================================================================
def build(args):
    num_classes = 151 if args.dataset != "oi" else 289
    num_rel_classes = 51 if args.dataset != "oi" else 31

    device = torch.device(args.device)
    backbone = build_backbone(args)

    model = PQDDSGG(
        backbone,
        num_classes=num_classes,
        num_rel_classes=num_rel_classes,
        num_rel_queries=args.num_rel_queries,
        hidden_dim=args.hidden_dim,
        nheads=args.nheads,
        enc_layers=args.enc_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        t1=args.t1,
        t2=args.t2,
        use_sgq=not args.no_sgq,
    )

    matchers = build_matchers(args)

    # loss weights (Section 3.5 of the paper)
    weight_dict = {
        "loss_cls_o2m": args.cls_loss_coef, "loss_rel_o2m": args.cls_loss_coef,
        "loss_bbox_o2m": args.bbox_loss_coef, "loss_giou_o2m": args.giou_loss_coef,
        "loss_cls_o2o": args.cls_loss_coef, "loss_rel_o2o": args.cls_loss_coef,
        "loss_bbox_o2o": args.bbox_loss_coef, "loss_giou_o2o": args.giou_loss_coef,
        "loss_cls_enc": args.aux_loss_coef, "loss_bbox_enc": args.aux_loss_coef,
        "loss_giou_enc": args.aux_loss_coef,
    }

    criterion = SetCriterionPQDD(num_classes, num_rel_classes, matchers,
                                 weight_dict=weight_dict, eos_coef=args.eos_coef)
    criterion.to(device)
    postprocessors = {"bbox": PostProcess()}
    return model, criterion, postprocessors
