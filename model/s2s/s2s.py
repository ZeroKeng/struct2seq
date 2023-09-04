from turtle import forward
import torch
import torch.nn as nn
from model.net.cotrm import CoTRM, CoTRMLayer
import numpy as np

class S2S(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.seq_emb = nn.Embedding(
            num_embeddings=cfg.n_vocab,
            embedding_dim=cfg.COTRM.d_model
        )
        self.struct_inproj = nn.Linear(cfg.d_struct,cfg.COTRM.d_model)
        #self.struct_outproj = nn.Linear(cfg.COTRM.d_model,cfg.d_struct)
        self.cotrm = CoTRMLayer(cfg = cfg)
        #self.pe
        self.pe = PositionalEncodings(max_len=1000, d_model=cfg.COTRM.d_model)
        
        self.TRMEncoderLayer = nn.TransformerEncoderLayer(
            d_model=cfg.COTRM.d_model,
            nhead=cfg.COTRM.nhead,
            dim_feedforward=cfg.COTRM.d_ffn,
            dropout=cfg.COTRM.dropout,
            batch_first=True
        )
    
    def forward(self, struct, seq, 
                struct_mask=None, seq_mask=None,
                struct_padding_mask=None, 
                seq_padding_mask=None):
        struct = self.struct_inproj(struct)
        seq = self.seq_emb(seq)
        
        struct = self.pe(struct)
        seq = self.pe(seq)
        seq = self.TRMEncoderLayer(seq,seq_mask,seq_padding_mask)
        struct, seq, _, _, _, _ =  self.cotrm(struct, seq, struct_mask, seq_mask, struct_padding_mask, seq_padding_mask)
        return struct, seq
    
class SeqRecover(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.s2s = S2S(cfg=cfg.S2S)
        self.TRMEncoderLayer = nn.TransformerDecoderLayer(
            d_model=cfg.SeqRecover.d_model,
            nhead=cfg.SeqRecover.nhead,
            dim_feedforward=cfg.SeqRecover.d_ffn,
            dropout=cfg.SeqRecover.dropout,
            batch_first=True
        )
        self.mask_ratio = cfg.SeqRecover.mask_ratio
        self.RecoverDecoder = nn.TransformerDecoder(
            self.TRMEncoderLayer,
            num_layers=cfg.SeqRecover.nlayer
        )
        self.outproj = nn.Linear(cfg.SeqRecover.d_model, cfg.S2S.n_vocab)
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.pe = PositionalEncodings(max_len=1000,d_model=cfg.SeqRecover.d_model)
        self.LayerNorm = nn.LayerNorm((cfg.SeqRecover.d_model,))
        
    def get_seq_mask(self, seq):
        seq_mask = torch.rand((seq.shape[1])).ge(self.mask_ratio)
        seq_mask = seq_mask.repeat((seq.shape[1],1))
        return seq_mask
    
    #load
    def load_s2s(self, path):
        self.s2s.load_state_dict(torch.load(path))

    #save
    def save_s2s(self, path):
        torch.save(self.s2s.state_dict(),path)
        
    def forward(self, struct, seq, struct_mask, struct_padding_mask, seq_padding_mask):
        seq_mask = self.get_seq_mask(seq)
        seq_ = seq.clone()
        seq_[:,seq_mask[0]] = 0
        struct, seq = self.s2s.forward(struct, seq_, struct_mask, seq_mask, struct_padding_mask, seq_padding_mask)
        # struct_seq = torch.concat((struct,seq), dim=1)
        # struct_seq = self.RecoverDecoder(struct_seq)
        seq = self.pe(seq)

        #return self.outproj(struct_seq[:,-seq.shape[1]:,:])
        seq = self.LayerNorm(seq)
        return self.outproj(seq)
        
    def compute_loss(self, struct, seq, seq_real, struct_mask, struct_padding_mask, seq_padding_mask):
        seq_recovered = self.forward(struct, seq, struct_mask, struct_padding_mask, seq_padding_mask)
        # [B,L,21]
        seq_recovered = seq_recovered.reshape(seq_recovered.shape[0] * seq_recovered.shape[1], -1)
        seq_real = seq_real.reshape(seq_real.shape[0] * seq_real.shape[1])
        loss = self.CrossEntropy.forward(seq_recovered,seq_real)
        return loss
    
class StructRecover(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.s2s = S2S(cfg=cfg.S2S)
        self.TRMEncoderLayer = nn.TransformerDecoderLayer(
            d_model=cfg.StructRecover.d_model,
            nhead=cfg.StructRecover.nhead,
            dim_feedforward=cfg.StructRecover.d_ffn,
            dropout=cfg.StructRecover.dropout,
            batch_first=True
        )
        self.mask_ratio = cfg.StructRecover.mask_ratio
        self.RecoverDecoder = nn.TransformerDecoder(
            self.TRMEncoderLayer,
            num_layers=cfg.StructRecover.nlayer
        )
        self.outproj = nn.Linear(cfg.StructRecover.d_model, cfg.S2S.d_struct)
        #non-linear
        self.CrossEntropy = nn.CrossEntropyLoss()
        self.MSELoss = nn.MSELoss()
        
    def get_struct_mask(self, struct):
        struct_mask = torch.rand((struct.shape[1])).ge(self.mask_ratio)
        struct_mask = struct_mask.repeat((struct.shape[1],1))
        #return None
        return struct_mask
    
    #load
    def load_s2s(self, path):
        self.s2s.load_state_dict(torch.load(path))

    #save
    def save_s2s(self, path):
        torch.save(self.s2s.state_dict(),path)
        
    def forward(self, struct, seq, seq_mask, struct_padding_mask, seq_padding_mask):
        struct_mask = self.get_struct_mask(struct)
        struct_ = struct.clone()
        struct_[:,struct_mask[0]] = 0
        struct, seq = self.s2s.forward(struct_, seq, struct_mask, seq_mask, struct_padding_mask, seq_padding_mask)
        # struct_seq = torch.concat((struct,seq), dim=1)
        # struct_seq = self.RecoverDecoder(struct_seq)
        struct = self.RecoverDecoder.forward(struct,seq)
        #return self.outproj(struct_seq[:,-seq.shape[1]:,:])
        return self.outproj(struct)
        
    def compute_loss(self, struct, seq, seq_mask, struct_padding_mask, seq_padding_mask):
        struct_recovered = self.forward(struct, seq, seq_mask, struct_padding_mask, seq_padding_mask)
        # [B,L,21]
        #seq_recovered = seq_recovered.reshape(seq_recovered.shape[0] * seq_recovered.shape[1], -1)
        loss = self.MSELoss.forward(struct_recovered,struct)
        return loss
    
class PositionalEncodings(nn.Module):
    def __init__(self,max_len,d_model):
        super(PositionalEncodings,self).__init__()
        self.max_len = max_len
        self.d_model = d_model
        
    def forward(self,X):
        pe = torch.zeros(self.max_len,self.d_model) #[ML,D]
        position = torch.arange(0, self.max_len).unsqueeze(1) #[ML,1]
        coef_sin = torch.exp(
            -(np.log(10000)/self.d_model) * torch.arange(0,self.d_model,2) #[ML/2]
        )
        coef_cos = torch.exp(
            -(np.log(10000)/self.d_model) * torch.arange(1,self.d_model,2) #[ML/2]
        )
        pe[:, 0::2] = torch.sin(position * coef_sin)
        pe[:, 1::2] = torch.cos(position * coef_cos)
        pe = pe.unsqueeze(0)
        with torch.no_grad():
            X = X + pe[:,:X.size(1),:] #X is [B,L,D] pe is [1,L,D,]
        return X