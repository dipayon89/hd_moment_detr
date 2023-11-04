# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import GroupNorm


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

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=2,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # TransformerEncoderLayerThin
        seft_atten_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                   dropout, activation, normalize_before)
        cross_atten_layer = CrossAttentionLayer(d_model, nhead,
                                                dropout, activation)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = VTCrossTransformerEncoder(seft_atten_layer, cross_atten_layer, num_encoder_layers, encoder_norm,
                                                 hidden_dim=d_model)

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


class VTCrossTransformerEncoder(nn.Module):

    def __init__(self, self_attn_layer, cross_attn_layer, num_layers=2, norm=None, return_intermediate=False,
                 hidden_dim=512):
        super().__init__()
        self.layers_vid = _get_clones(self_attn_layer, num_layers)
        self.layers_txt = _get_clones(self_attn_layer, num_layers)
        self.layers_cross = cross_attn_layer
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

        for layer in self.layers_vid:
            output_vid = layer(output_vid, src_key_padding_mask=src_vid_key_padding_mask, pos=pos_vid)

        for layer in self.layers_txt:
            output_txt = layer(output_txt, src_key_padding_mask=src_txt_key_padding_mask, pos=pos_txt)

        # output = torch.cat([output_vid, output_txt], dim=0)
        # pos = torch.cat([pos_vid, pos_txt], dim=0)
        # src_key_padding_mask = torch.cat([src_vid_key_padding_mask, src_txt_key_padding_mask], dim=1)
        output = self.layers_cross(output_vid, output_txt,
                                   src1_key_padding_mask=src_vid_key_padding_mask,
                                   src2_key_padding_mask=src_txt_key_padding_mask,
                                   pos1=pos_vid, pos2=pos_txt)
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
    def __init__(self, d_model, nhead, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.pooling = nn.AvgPool1d(3, stride=2, padding=1)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.linear3 = nn.Linear(d_model, d_model)
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


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return VTCrossTransformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
