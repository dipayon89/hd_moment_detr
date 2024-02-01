# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .attention import MultiheadAttention


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # TransformerDecoderLayerThin
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)  # (L, batch_size, d)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)  # (#layers, #queries, batch_size, d)
        hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        memory = memory.transpose(0, 1)  # (batch_size, L, d)
        return hs, memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class VTTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = VTTransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm, hidden_dim=d_model)

        # TransformerDecoderLayerThin
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_vid, src_txt, vid_mask, txt_mask, pos_embed_vid, pos_embed_txt, query_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src_vid.shape
        src_vid = src_vid.permute(1, 0, 2)  # (L, batch_size, d)
        src_txt = src_txt.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed_vid = pos_embed_vid.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed_txt = pos_embed_txt.permute(1, 0, 2)  # (L, batch_size, d)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        memory, pos_embed, mask = self.encoder(src_vid, src_txt,
                                               src_vid_key_padding_mask=vid_mask,
                                               src_txt_key_padding_mask=txt_mask,
                                               pos_vid=pos_embed_vid,
                                               pos_txt=pos_embed_txt)  # (L, batch_size, d)

        memory_global, memory_local = memory[0], memory[1:]
        mask_local = mask[:, 1:]
        pos_embed_local = pos_embed[1:]

        tgt = torch.zeros_like(query_embed, device=src_vid.device)
        hs = self.decoder(tgt, memory_local, memory_key_padding_mask=mask_local,
                          pos=pos_embed_local, query_pos=query_embed)  # (#layers, #queries, batch_size, d)
        hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        memory = memory_local.transpose(0, 1)  # (batch_size, L, d)
        return hs, memory, memory_global


class VTTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False, hidden_dim=512):
        super().__init__()
        self.layers_vid = _get_clones(encoder_layer, num_layers)
        self.layers_txt = _get_clones(encoder_layer, num_layers)
        self.layers_cross = _get_clones(encoder_layer, num_layers * 2)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, src_vid, src_txt,
                src_vid_key_padding_mask: Optional[Tensor] = None,
                src_txt_key_padding_mask: Optional[Tensor] = None,
                pos_vid: Optional[Tensor] = None,
                pos_txt: Optional[Tensor] = None):

        output_vid = src_vid
        output_txt = src_txt

        intermediate = []

        for layer in self.layers_vid:
            output_vid = layer(output_vid, src_key_padding_mask=src_vid_key_padding_mask, pos=pos_vid)

        for layer in self.layers_txt:
            output_txt = layer(output_txt, src_key_padding_mask=src_txt_key_padding_mask, pos=pos_txt)

        output = torch.cat([output_vid, output_txt], dim=0)
        pos = torch.cat([pos_vid, pos_txt], dim=0)
        src_key_padding_mask = torch.cat([src_vid_key_padding_mask, src_txt_key_padding_mask], dim=1)

        # print("output.shape", output.shape)
        # print("pos.shape", pos.shape)
        # print("src_key_padding_mask.shape", src_key_padding_mask.shape)

        # for global token
        output = output.permute(1, 0, 2)  # (batch_size, L, d)
        output_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(output.shape[0], 1, 1)
        output = torch.cat([output_, output], dim=1)
        output = output.permute(1, 0, 2)  # (L, batch_size, d)
        # print("new output.shape", output.shape)

        pos = pos.permute(1, 0, 2)  # (batch_size, L, d)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)
        pos = pos.permute(1, 0, 2)  # (L, batch_size, d)
        # print("new pos.shape", pos.shape)

        src_key_padding_mask_ = torch.tensor([[True]]).to(src_key_padding_mask.device).repeat(
            src_key_padding_mask.shape[0], 1)
        src_key_padding_mask = torch.cat([src_key_padding_mask_, src_key_padding_mask], dim=1)
        # print("new src_key_padding_mask.shape", src_key_padding_mask.shape)

        for layer in self.layers_cross:
            output = layer(output, src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, pos, src_key_padding_mask


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class VTCrossTransformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_queries=2, num_encoder_layers=2,
                 num_decoder_layers=6, dim_feedforward=2048, output_dim = 75,  dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, query_dim=2,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 num_patterns=0,
                 modulate_t_attn=True,
                 bbox_embed_diff_each_layer=False, ):
        super().__init__()

        # TransformerEncoderLayerThin
        encoder = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                          dropout, activation, normalize_before)
        seft_atten_layer = PoolformerLayer(d_model, dim_feedforward, dropout, activation)
        cross_atten_layer = CrossAttentionLayer(d_model, nhead, dim_feedforward,
                                                dropout=dropout, activation=activation)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = VTCrossTransformerEncoder(seft_atten_layer, cross_atten_layer, encoder, num_encoder_layers,
                                                 encoder_norm,
                                                 hidden_dim=d_model)

        # TransformerDecoderLayerThin
        # decoder_layer = VTTransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                           dropout, activation, normalize_before, keep_query_pos=keep_query_pos)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = VTTransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                     return_intermediate=return_intermediate_dec,
        #                                     d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos,
        #                                     query_scale_type=query_scale_type,
        #                                     modulate_t_attn=modulate_t_attn,
        #                                     bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self.class_estimator = ClassPredictionHead(d_model, num_queries)
        self.localization_estimator = LocalizationPredictionHead(d_model, num_queries, activation=activation)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        self.num_patterns = num_patterns
        self.global_threshold = LearnableThreshold()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src_vid, src_txt, vid_mask, txt_mask, pos_embed_vid, pos_embed_txt, query_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src_vid.shape
        src_vid = src_vid.permute(1, 0, 2)  # (L, batch_size, d)
        src_txt = src_txt.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed_vid = pos_embed_vid.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed_txt = pos_embed_txt.permute(1, 0, 2)  # (L, batch_size, d)
        refpoint_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        memory, pos_embed, mask = self.encoder(src_vid, src_txt,
                                               src_vid_key_padding_mask=vid_mask,
                                               src_txt_key_padding_mask=txt_mask,
                                               pos_vid=pos_embed_vid,
                                               pos_txt=pos_embed_txt)  # (L, batch_size, d)

        memory_global, memory_local = memory[0], memory[1:]
        # mask_local = mask[:, 1:]
        # pos_embed_local = pos_embed[1:]

        # tgt = torch.zeros(refpoint_embed.shape[0], bs, d, device=src_vid.device)
        # hs, references = self.decoder(tgt, memory_local, memory_key_padding_mask=mask_local,
        #                               pos=pos_embed_local,
        #                               refpoints_unsigmoid=refpoint_embed)  # (#layers, #queries, batch_size, d)

        hs = self.class_estimator(memory_global)
        references = self.localization_estimator(memory_global)

        memory_local = memory_local.transpose(0, 1)  # (batch_size, L, d)
        return hs, references, memory_local, self.global_threshold(memory_global)


class VTCrossTransformerEncoder(nn.Module):

    def __init__(self, self_attn_layer, cross_attn_layer, encoder, num_layers=2, norm=None, return_intermediate=False,
                 hidden_dim=512,
                 apply_pooler_before_mixing=False,
                 apply_self_attention_after_mixing=True):
        super().__init__()
        self.apply_pooler_before_mixing = apply_pooler_before_mixing
        self.apply_self_attention_after_mixing = apply_self_attention_after_mixing
        if self.apply_pooler_before_mixing:
            self.layers_vid = _get_clones(self_attn_layer, 2)
            self.layers_txt = _get_clones(self_attn_layer, 2)

        self.layers_cross = cross_attn_layer

        if self.apply_self_attention_after_mixing:
            self.layers_encoder = _get_clones(encoder, num_layers)

        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))
        self.hidden_dim = hidden_dim

    def forward(self, src_vid, src_txt,
                src_vid_key_padding_mask: Optional[Tensor] = None,
                src_txt_key_padding_mask: Optional[Tensor] = None,
                pos_vid: Optional[Tensor] = None,
                pos_txt: Optional[Tensor] = None, ):
        # print("output.shape", output.shape)
        # print("pos.shape", pos.shape)
        # print("src_key_padding_mask.shape", src_key_padding_mask.shape)

        # for global token
        src_vid = src_vid.permute(1, 0, 2)  # (batch_size, L, d)
        src_vid_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_vid.shape[0], 1, 1)
        src_vid = torch.cat([src_vid_, src_vid], dim=1)
        src_vid = src_vid.permute(1, 0, 2)  # (L, batch_size, d)
        # print("new output.shape", output.shape)

        pos_vid = pos_vid.permute(1, 0, 2)  # (batch_size, L, d)
        pos_vid_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_vid.shape[0], 1, 1)
        pos_vid = torch.cat([pos_vid_, pos_vid], dim=1)
        pos_vid = pos_vid.permute(1, 0, 2)  # (L, batch_size, d)
        # print("new pos.shape", pos.shape)

        src_vid_key_padding_mask_ = torch.tensor([[True]]).to(src_vid_key_padding_mask.device).repeat(
            src_vid_key_padding_mask.shape[0], 1)
        src_vid_key_padding_mask = torch.cat([src_vid_key_padding_mask_, src_vid_key_padding_mask], dim=1)
        # print("new src_key_padding_mask.shape", src_key_padding_mask.shape)

        output_vid = src_vid
        output_txt = src_txt

        intermediate = []

        # for layer in self.layers_vid:
        #     output_vid = layer(output_vid, src_key_padding_mask=src_vid_key_padding_mask, pos=pos_vid)
        #
        # for layer in self.layers_txt:
        #     output_txt = layer(output_txt, src_key_padding_mask=src_txt_key_padding_mask, pos=pos_txt)

        if self.apply_pooler_before_mixing:
            for layer in self.layers_vid:
                output_vid = layer(output_vid, pos=pos_vid)

            for layer in self.layers_txt:
                output_txt = layer(output_txt, pos=pos_txt)

        # output = torch.cat([output_vid, output_txt], dim=0)
        # pos = torch.cat([pos_vid, pos_txt], dim=0)
        # src_key_padding_mask = torch.cat([src_vid_key_padding_mask, src_txt_key_padding_mask], dim=1)

        # mixing avToken before self attention
        output = self.layers_cross(output_vid, output_txt,
                                   src1_key_padding_mask=src_vid_key_padding_mask,
                                   src2_key_padding_mask=src_txt_key_padding_mask,
                                   pos1=pos_vid, pos2=pos_txt)
        if self.return_intermediate:
            intermediate.append(output)

        if self.apply_self_attention_after_mixing:
            for layer in self.layers_encoder:
                output = layer(output,
                               src_key_padding_mask=src_vid_key_padding_mask, pos=pos_vid)
                if self.return_intermediate:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output, pos_vid, src_vid_key_padding_mask


class PoolformerLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.pooling_layer = nn.AvgPool1d(3, stride=1, padding=1)
        # self.pooling_layer = nn.MaxPool1d(3, stride=1, padding=1)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos: Optional[Tensor] = None):
        src = self.with_pos_embed(src, pos)
        src2 = self.pooling_layer(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.pooling = nn.AvgPool1d(3, stride=2, padding=1)
        self.pooling = nn.MaxPool1d(3, stride=2, padding=1)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, dim_feedforward)
        self.linear3 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, src1, src2,
                src1_mask: Optional[Tensor] = None,
                src2_mask: Optional[Tensor] = None,
                src1_key_padding_mask: Optional[Tensor] = None,
                src2_key_padding_mask: Optional[Tensor] = None,
                pos1: Optional[Tensor] = None,
                pos2: Optional[Tensor] = None):
        k = v = self.with_pos_embed(src2, pos2)
        out1 = self.cross_attn_1(query=src1, key=k, value=v, attn_mask=src2_mask,
                                 key_padding_mask=src2_key_padding_mask)[0]
        out1 = self.norm1(self.linear1(self.dropout1(out1)))
        # print("out1.shape", out1.shape)
        out2 = self.self_attn(query=out1, key=out1, value=out1, attn_mask=src2_mask,
                              key_padding_mask=src1_key_padding_mask)[0]
        out2 = self.dropout2(out2)

        # print("out2.shape", out2.shape)
        out3 = self.norm2(self.pooling(torch.cat([out1, out2], dim=2)))

        # print("out3.shape", out3.shape)
        k = v = self.with_pos_embed(src1, pos1)

        out = self.cross_attn_2(query=out3, key=k, value=v, attn_mask=src1_mask,
                                key_padding_mask=src1_key_padding_mask)[0]
        # print("out.shape", out.shape)
        out1 = self.linear3(self.dropout3(self.activation(self.linear2(out))))
        out = out + self.dropout4(out1)
        out = self.norm3(out)
        # print("out.shape", out.shape)
        return out


class VTTransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, query_dim=2, keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_t_attn=False,
                 bbox_embed_diff_each_layer=False,
                 ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

        self.ref_point_head = MLP(d_model, d_model, d_model, 2)

        # self.bbox_embed = None
        # for DAB-deter
        if bbox_embed_diff_each_layer:
            self.bbox_embed = nn.ModuleList([MLP(d_model, d_model, 2, 3) for i in range(num_layers)])
        else:
            self.bbox_embed = MLP(d_model, d_model, 2, 3)
        # init bbox_embed
        if bbox_embed_diff_each_layer:
            for bbox_embed in self.bbox_embed:
                nn.init.constant_(bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(bbox_embed.layers[-1].bias.data, 0)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        self.d_model = d_model
        self.modulate_t_attn = modulate_t_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        if modulate_t_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 1, 2)

        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                ):
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        # import ipdb; ipdb.set_trace()

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :self.query_dim]
            # get sine embedding for the query vector
            # print('obj_center', obj_center.shape)
            query_sine_embed = gen_sineembed_for_position(obj_center)
            # print('line230', query_sine_embed.shape)
            query_pos = self.ref_point_head(query_sine_embed)
            # print('line232',query_sine_embed.shape)
            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            # print(query_sine_embed.shape) # 10 32 512
            query_sine_embed = query_sine_embed * pos_transformation

            # modulated HW attentions
            if self.modulate_t_attn:
                reft_cond = self.ref_anchor_head(output).sigmoid()  # nq, bs, 1
                # print(reft_cond.shape, reft_cond[..., 0].shape) # 10 32 1, 10 32
                # print(obj_center.shape, obj_center[..., 1].shape) # 10 32 2, 10 32
                # print(query_sine_embed.shape) # 10 32 256

                query_sine_embed *= (reft_cond[..., 0] / obj_center[..., 1]).unsqueeze(-1)

            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0))

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](output)
                else:
                    tmp = self.bbox_embed(output)
                # import ipdb; ipdb.set_trace()
                # print("reference_points.shape", reference_points.shape)
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points)
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2)
                ]

        return output.unsqueeze(0)


class VTTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 2, nhead, dropout=dropout, vdim=d_model)

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                is_first=False):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            q_content = self.sa_qcontent_proj(tgt)  # target is the input of the first decoder layer. zero by default.
            q_pos = self.sa_qpos_proj(query_pos)
            k_content = self.sa_kcontent_proj(tgt)
            k_pos = self.sa_kpos_proj(query_pos)
            v = self.sa_v_proj(tgt)

            num_queries, bs, n_model = q_content.shape
            hw, _, _ = k_content.shape

            q = q_content + q_pos
            k = k_content + k_pos

            tgt2 = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)[0]
            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3).view(num_queries, bs, n_model * 2)
        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, n_model * 2)

        tgt2 = self.cross_attn(query=q,
                               key=k,
                               value=v, attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


class ClassPredictionHead(nn.Module):
    """ Simple Prediction Head consisting of a conv layer and a linear layer """

    def __init__(self, d_model, out_dim,
                 in_channel=1, out_channel=2,
                 num_forward_conv_layer=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.conv_forward = nn.ModuleList(
            [copy.deepcopy(nn.Conv1d(in_channel, d_model, kernel_size=2*(i+1)+1, padding=i+1)) for i in range(num_forward_conv_layer)])
        self.conv_backward = nn.Conv1d(d_model, out_channel, kernel_size=5, padding=2)
        self.linear = nn.Linear(d_model, out_dim)
        self.activation = LearnableThreshold(0.1)
        self.dropout = dropout

    def forward(self, mixed_data, ):
        x = mixed_data.unsqueeze(dim=0)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        x1 = torch.zeros((x.shape[0],self.d_model,x.shape[2]), device=x.device, dtype=x.dtype)
        for conv_id, conv in enumerate(self.conv_forward):
            x1 += conv(F.dropout(x, p=self.dropout))
        x = self.conv_backward(x1)
        x = x.squeeze(dim=1)
        x = F.dropout(x, p=self.dropout)
        x = self.linear(self.activation(x))
        x = x.permute(0, 2, 1)
        return x.unsqueeze(0)


class LocalizationPredictionHead(nn.Module):
    """ Simple Prediction Head consisting of a conv layer and a linear layer """

    def __init__(self, d_model, out_dim,
                 in_channel=1, out_channel=2,
                 num_forward_conv_layer=4,
                 dropout=0.1, activation="relu"):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.conv_forward = nn.ModuleList(
            [copy.deepcopy(nn.Conv1d(in_channel, d_model, kernel_size=2 * (i + 1) + 1, padding=i + 1)) for i in
             range(num_forward_conv_layer)])
        self.conv_backward = nn.Conv1d(d_model, out_channel, kernel_size=5, padding=2)
        self.linear = nn.Linear(d_model, out_dim)
        self.activation = _get_activation_fn(activation)
        self.dropout = dropout

    def forward(self, mixed_data):
        x = mixed_data.unsqueeze(dim=0)
        x = x.permute(1, 0, 2)
        x = self.norm(x)
        x1 = torch.zeros((x.shape[0], self.d_model, x.shape[2]), device=x.device, dtype=x.dtype)
        for conv_id, conv in enumerate(self.conv_forward):
            x1 += conv(F.dropout(x, p=self.dropout))
        x = self.conv_backward(x1)
        x = x.squeeze(dim=1)
        x = F.dropout(x, p=self.dropout)
        x = self.linear(x)
        x = self.activation(x)
        x = x.permute(0, 2, 1)
        # x = self.linear2(x)
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

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    center_embed = pos_tensor[:, :, 0] * scale
    pos_x = center_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)

    span_embed = pos_tensor[:, :, 1] * scale
    pos_w = span_embed[:, :, None] / dim_t
    pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

    pos = torch.cat((pos_x, pos_w), dim=2)
    return pos


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return VTCrossTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
        activation='prelu',
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.prelu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
