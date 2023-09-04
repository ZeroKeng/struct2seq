from turtle import forward
from typing import Callable, Union, Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F 
from omegaconf import DictConfig, OmegaConf

#cfg = OmegaConf.load('./config/s2s.yamls')
#TRM input:[B,L,d_model] output: [B,L,d_model]

class CoTRMLayerHalf(nn.TransformerDecoderLayer):
    """ Co-transformer based on transformer decoderlayer.
    """
    def __init__(self, 
                 d_model: int, 
                 nhead: int, 
                 dim_feedforward: int = 2048, 
                 dropout: float = 0.1, 
                 activation: Callable[[Tensor], Tensor] = F.relu, 
                 layer_norm_eps: float = 0.00001, 
                 batch_first: bool = True, 
                 norm_first: bool = False, 
                 device=None, 
                 dtype=None) -> None:
        super().__init__(d_model,
                         nhead,
                         dim_feedforward,
                         dropout, activation,
                         layer_norm_eps,
                         batch_first,
                         norm_first,
                         device,
                         dtype)
    def forward(self, 
                query: Tensor, 
                key_value: Tensor, 
                query_mask: Optional[Tensor] = None, 
                key_value_mask: Optional[Tensor] = None,
                query_key_padding_mask: Optional[Tensor] = None, 
                key_value_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            query: the sequence to the decoder layer (required).
            key_value: the sequence from the last layer of the encoder (required).
            query_mask: the mask for the query sequence (optional).
            key_value_mask: the mask for the key_value sequence (optional).
            query_key_padding_mask: the mask for the query keys per batch (optional).
            key_value_key_padding_mask: the mask for the key_value keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = query
        #print('x size is:', x.shape)
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), query_mask, query_key_padding_mask)
            x = x + self._mha_block(self.norm2(x), key_value, key_value_mask, key_value_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, query_mask, query_key_padding_mask))
            x = self.norm2(x + self._mha_block(x, key_value, key_value_mask, key_value_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x
    
        # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)



class CoTRMLayer(nn.Module):
    def __init__(self, cfg):
        super(CoTRMLayer, self).__init__()
        self.cfg = cfg
        self.LeftHalf = CoTRMLayerHalf(
            d_model=cfg.COTRM.d_model,
            nhead = cfg.COTRM.nhead,
            dim_feedforward = cfg.COTRM.d_ffn,
            dropout = cfg.COTRM.dropout
        )
        self.RightHalf = CoTRMLayerHalf(
            d_model=cfg.COTRM.d_model,
            nhead = cfg.COTRM.nhead,
            dim_feedforward = cfg.COTRM.d_ffn,
            dropout = cfg.COTRM.dropout,
        )
        self.LeftTRM = nn.TransformerEncoderLayer(
            d_model=cfg.COTRM.TRM.d_model,
            nhead = cfg.COTRM.TRM.nhead,
            dim_feedforward = cfg.COTRM.TRM.d_ffn,
            dropout = cfg.COTRM.TRM.dropout,
            batch_first=True
        )
        self.RightTRM = nn.TransformerEncoderLayer(
            d_model=cfg.COTRM.TRM.d_model,
            nhead = cfg.COTRM.TRM.nhead,
            dim_feedforward = cfg.COTRM.TRM.d_ffn,
            dropout = cfg.COTRM.TRM.dropout,
            batch_first=True
        )

    def forward(self, struct, seq, 
                struct_mask=None, seq_mask=None,
                struct_padding_mask=None, 
                seq_padding_mask=None):
        struct_hidden = self.LeftHalf.forward(
            query = struct,
            key_value = seq,
            query_mask=struct_mask,
            query_key_padding_mask=struct_padding_mask,
            key_value_key_padding_mask=seq_padding_mask
        )
        seq_hidden = self.RightHalf.forward(
            query = seq,
            key_value = struct,
            query_mask=seq_mask,
            query_key_padding_mask=seq_padding_mask,
            key_value_key_padding_mask=struct_padding_mask
        )
        struct_hidden = self.LeftTRM.forward(
            src=struct_hidden,
            src_mask=struct_mask,
            src_key_padding_mask=struct_padding_mask
        )
        seq_hidden = self.RightTRM.forward(
            src=seq_hidden,
            src_mask=seq_mask,
            src_key_padding_mask=seq_padding_mask
        )
        
        return struct_hidden, seq_hidden, struct_mask, seq_mask, struct_padding_mask, seq_padding_mask

class CoTRM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.CoTRMLayers = [CoTRMLayer(cfg=cfg) for i in range(cfg.COTRM.nlayer)]
    
    def forward(self, *input):
        for layer in self.CoTRMLayers:
            input = layer.forward(*input)
        return input