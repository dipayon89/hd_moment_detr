# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from moment_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from moment_detr.matcher import build_matcher
from moment_detr.transformer import build_transformer
from moment_detr.position_encoding import build_position_encoding
from moment_detr.misc import accuracy


class MomentDETR(nn.Module):
    """ This is the Moment-DETR module that performs moment localization. """

    def __init__(self, transformer, position_embed, txt_position_embed, txt_dim, vid_dim,
                 num_queries, input_dropout, aux_loss=False,
                 contrastive_align_loss=False, contrastive_hdim=64,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture. See transformer.py
            position_embed: torch module of the position_embedding, See position_encoding.py
            txt_position_embed: position_embedding for text
            txt_dim: int, text query input dimension
            vid_dim: int, video feature input dimension
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Moment-DETR can detect in a single video.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            contrastive_align_loss: If true, perform span - tokens contrastive learning
            contrastive_hdim: dimension used for projecting the embeddings before computing contrastive loss
            max_v_l: int, maximum #clips in videos
            span_loss_type: str, one of [l1, ce]
                l1: (center-x, width) regression.
                ce: (st_idx, ed_idx) classification.
            # foreground_thd: float, intersection over prediction >= foreground_thd: labeled as foreground
            # background_thd: float, intersection over prediction <= background_thd: labeled background
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        hidden_dim = transformer.d_model
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        # self.span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        self.span_embed = SpanPredictionHead(hidden_dim, span_pred_dim=span_pred_dim, in_channel=num_queries,
                                             out_channel=num_queries)
        # self.class_embed = nn.Linear(hidden_dim, 2)  # 0: background, 1: foreground
        self.class_embed = ClassPredictionHead(hidden_dim, num_class=2, in_channel=num_queries,
                                               out_channel=num_queries)  # 0: background, 1: foreground
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj
        # self.foreground_thd = foreground_thd
        # self.background_thd = background_thd
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        relu_args = [True] * 3
        relu_args[n_input_proj - 1] = False
        self.input_txt_proj = nn.Sequential(*[
                                                 LinearLayer(txt_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[0]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[1]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[2])
                                             ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
                                                 LinearLayer(vid_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[0]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[1]),
                                                 LinearLayer(hidden_dim, hidden_dim, layer_norm=True,
                                                             dropout=input_dropout, relu=relu_args[2])
                                             ][:n_input_proj])
        # self.input_txt_proj = PreprocessingModule(txt_dim, hidden_dim)
        # self.input_vid_proj = PreprocessingModule(vid_dim, hidden_dim)

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)

        # self.saliency_proj = nn.Linear(hidden_dim, 1)
        # self.saliency_proj = ClassPredictionHead(hidden_dim, num_class=1, in_channel=75, out_channel=75)
        # self.saliency_proj = CustomLinearLayer(hidden_dim, 1)
        self.saliency_proj1 = CustomLinearLayer(hidden_dim, hidden_dim)
        self.saliency_proj2 = CustomLinearLayer(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.aux_loss = aux_loss

    def forward(self, src_txt, src_txt_mask, src_vid, src_vid_mask):
        """The forward expects two tensors:
               - src_txt: [batch_size, L_txt, D_txt]
               - src_txt_mask: [batch_size, L_txt], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer
               - src_vid: [batch_size, L_vid, D_vid]
               - src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
                    will convert to 1 as padding later for transformer

            It returns a dict with the following elements:
               - "pred_spans": The normalized boxes coordinates for all queries, represented as
                               (center_x, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)
        # TODO should we remove or use different positional embeddings to the src_txt?
        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)
        # pos_txt = torch.zeros_like(src_txt)
        # pad zeros for txt positions
        pos = torch.cat([pos_vid, pos_txt], dim=1)
        # (#layers, bsz, #queries, d), (bsz, L_vid+L_txt, d)
        hs, memory, global_memory = self.transformer(src, ~mask, self.query_embed.weight, pos, self.max_v_l)
        outputs_class = self.class_embed(hs[-1])  # (#layers, batch_size, #queries, #classes)
        outputs_coord = self.span_embed(hs[-1])  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        # outputs_class = self.class_embed(hs)  # (#layers, batch_size, #queries, #classes)
        # outputs_coord = self.span_embed(hs)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord = outputs_coord.sigmoid()
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        # out["saliency_scores"] = self.saliency_proj(vid_mem).squeeze(-1).squeeze(0)  # (bsz, L_vid)
        out["saliency_scores"] = (
                torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(global_memory).unsqueeze(1),
                          dim=-1) / np.sqrt(self.hidden_dim))

        ### Neg Pairs ###
        src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
        src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()
        pos_neg = pos.clone()  # since it does not use actual content

        hs_neg, memory_neg, global_memory_neg = self.transformer(src_neg, ~mask_neg, self.query_embed.weight, pos_neg,
                                                                 self.max_v_l)
        outputs_class_neg = self.class_embed(hs_neg[-1])  # (#layers, batch_size, #queries, #classes)
        outputs_coord_neg = self.span_embed(hs_neg[-1])  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        # outputs_class_neg = self.class_embed(hs_neg)  # (#layers, batch_size, #queries, #classes)
        # outputs_coord_neg = self.span_embed(hs_neg)  # (#layers, bsz, #queries, 2 or max_v_l * 2)
        if self.span_loss_type == "l1":
            outputs_coord_neg = outputs_coord_neg.sigmoid()
        out.update({'pred_logits_neg': outputs_class_neg[-1], 'pred_spans_neg': outputs_coord_neg[-1]})

        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        out["saliency_scores_neg"] = (
                torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(global_memory_neg).unsqueeze(1),
                          dim=-1) / np.sqrt(self.hidden_dim))

        if self.aux_loss:
            # assert proj_queries and proj_txt_mem
            out['aux_outputs'] = [
                {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        return out

    # @torch.jit.unused
    # def _set_aux_loss(self, outputs_class, outputs_coord):
    #     # this is a workaround to make torchscript happy, as torchscript
    #     # doesn't support dictionary with non-homogeneous values, such
    #     # as a dict having both a Tensor and a list.
    #     return [{'pred_logits': a, 'pred_spans': b}
    #             for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, triplet_margin=5):
        """ Create the criterion.
        Parameters:
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
            temperature: float, temperature for NCE loss
            span_loss_type: str, [l1, ce]
            max_v_l: int,
            saliency_margin: float
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin
        self.triplet_margin = triplet_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')

            # giou
            # src_span_indices = src_spans.max(1)[1]  # (#spans, 2)
            # src_span_indices[:, 1] += 1  # ed non-inclusive [st, ed)
            #
            # tgt_span_indices = tgt_spans
            # tgt_span_indices[:, 1] += 1
            # loss_giou = 1 - torch.diag(generalized_temporal_iou(src_span_indices, tgt_span_indices))
            loss_giou = loss_span.new_zeros([1])

        losses = {'loss_giou': loss_giou.mean()}
        if "pred_spans_neg" in targets:
            neg_spans = outputs['pred_spans_neg'][idx]  # (#spans, max_v_l * 2)
            if self.span_loss_type == "l1":
                loss_span_triplet = F.triplet_margin_with_distance_loss(tgt_spans, src_spans, neg_spans,
                                                                        distance_function=F.l1_loss,
                                                                        margin=self.triplet_margin)
            else:
                loss_span_triplet = F.triplet_margin_with_distance_loss(tgt_spans, src_spans, neg_spans,
                                                                        distance_function=F.cross_entropy,
                                                                        margin=self.triplet_margin)
            losses['loss_span'] = loss_span.mean() + loss_span_triplet.mean()
        else:
            losses['loss_span'] = loss_span.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # TODO add foreground and background classifier.  use all non-matched as background.
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
        # idx is a tuple of two 1D tensors (batch_idx, src_idx), of the same length == #objects in batch
        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                    dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
        target_classes[idx] = self.foreground_label

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
        losses = {'loss_label': loss_ce.mean()}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]
        return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}
        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        if "highlighted_clips" not in targets:
            return {"loss_saliency": loss_saliency}

        highlighted_clips = targets["highlighted_clips"]
        saliency_scores_neg = outputs["saliency_scores_neg"]  # (N, L)

        return {"loss_saliency": loss_saliency + F.mse_loss(saliency_scores, highlighted_clips)
                                 + F.triplet_margin_with_distance_loss(highlighted_clips,
                                                                       saliency_scores,
                                                                       saliency_scores_neg,
                                                                       distance_function=F.mse_loss,
                                                                       margin=self.triplet_margin)}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # list(tuples), each tuple is (pred_span_indices, tgt_span_indices)
        indices = self.matcher(outputs_without_aux, targets)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PreprocessingModule(nn.Module):
    """ Simple Prediction Head consisting of a conv layer and a linear layer """

    def __init__(self, in_channel, out_channel,
                 num_forward_conv_layer=3, dropout=0.1):
        super().__init__()
        self.out_channel = out_channel
        self.d_model = out_channel * 2
        self.conv_forward = nn.ModuleList(
            [nn.Conv1d(in_channel, self.d_model, kernel_size=2 * (i + 1) + 1, padding=i + 1) for i in
             range(num_forward_conv_layer)])
        self.conv_backward = nn.Conv1d(self.d_model, out_channel, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(self.out_channel)
        self.dropout = dropout

    def forward(self, x, ):
        x = x.permute(0, 2, 1)
        x1 = torch.zeros((x.shape[0], self.d_model, x.shape[2]), device=x.device, dtype=x.dtype)
        for conv_id, conv in enumerate(self.conv_forward):
            x1 += conv(F.dropout(x, p=self.dropout))
        x = self.conv_backward(x1)
        x = x.permute(0, 2, 1)
        x = self.norm(F.relu(x, inplace=True))
        return x


class ClassPredictionHead(nn.Module):
    """ Simple Prediction Head consisting of a conv layer and a linear layer """

    def __init__(self, d_model, num_class=2,
                 in_channel=10, out_channel=10,
                 num_forward_conv_layer=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.conv_forward = nn.ModuleList(
            [nn.Conv1d(in_channel, d_model, kernel_size=2 * (i + 1) + 1, padding=i + 1) for i in
             range(num_forward_conv_layer)])
        self.conv_backward = nn.Conv1d(d_model, out_channel, kernel_size=5, padding=2)
        self.linear = nn.Linear(d_model, num_class)
        self.activation = LearnableThreshold(0.1)
        self.dropout = dropout

    def forward(self, mixed_data, ):
        x = self.norm(mixed_data)
        x1 = torch.zeros((x.shape[0], self.d_model, x.shape[2]), device=x.device, dtype=x.dtype)
        for conv_id, conv in enumerate(self.conv_forward):
            x1 += conv(F.dropout(x, p=self.dropout))
        x = self.conv_backward(x1)
        x = x.squeeze(dim=1)
        x = F.dropout(x, p=self.dropout)
        x = self.linear(self.activation(x))
        return x.unsqueeze(0)


class SpanPredictionHead(nn.Module):
    """ Simple Prediction Head consisting of a conv layer and a linear layer """

    def __init__(self, d_model, span_pred_dim=2,
                 in_channel=10, out_channel=10,
                 num_forward_conv_layer=3,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.conv_forward = nn.ModuleList(
            [nn.Conv1d(in_channel, d_model, kernel_size=2 * (i + 1) + 1, padding=i + 1) for i in
             range(num_forward_conv_layer)])
        self.conv_backward = nn.Conv1d(d_model, out_channel, kernel_size=5, padding=2)
        # self.linear = nn.Linear(d_model, span_pred_dim)
        self.mlp = MLP(d_model, d_model, span_pred_dim, 3)
        self.dropout = dropout

    def forward(self, mixed_data):
        x = self.norm(mixed_data)
        x1 = torch.zeros((x.shape[0], self.d_model, x.shape[2]), device=x.device, dtype=x.dtype)
        for conv_id, conv in enumerate(self.conv_forward):
            x1 += conv(F.dropout(x, p=self.dropout))
        x = self.conv_backward(x1)
        x = x.squeeze(dim=1)
        x = F.dropout(x, p=self.dropout)
        x = self.mlp(x)
        return x.unsqueeze(0)


class LearnableThreshold(nn.Module):

    def __init__(self, init: float = 0.25,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.init = init
        self.weight = nn.Parameter(torch.empty(1, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.weight, self.init)

    def forward(self, in_data: Tensor) -> Tensor:
        return F.threshold(in_data, self.weight.item(), 0)


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.relu = nn.PReLU()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = self.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class CustomLinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(CustomLinearLayer, self).__init__()
        self.relu = relu
        if self.relu:
            self.activation = LearnableThreshold(0.1)
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = self.activation(x)
        return x  # (N, L, D)


def build_model(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/moment_detr/issues/108#issuecomment-650269223
    device = torch.device(args.device)

    transformer = build_transformer(args)
    position_embedding, txt_position_embedding = build_position_encoding(args)

    model = MomentDETR(
        transformer,
        position_embedding,
        txt_position_embedding,
        txt_dim=args.t_feat_dim,
        vid_dim=args.v_feat_dim,
        num_queries=args.num_queries,
        input_dropout=args.input_dropout,
        aux_loss=args.aux_loss,
        contrastive_align_loss=args.contrastive_align_loss,
        contrastive_hdim=args.contrastive_hdim,
        span_loss_type=args.span_loss_type,
        use_txt_pos=args.use_txt_pos,
        n_input_proj=args.n_input_proj,
    )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
    criterion = SetCriterion(
        matcher=matcher, weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin
    )
    criterion.to(device)
    return model, criterion
