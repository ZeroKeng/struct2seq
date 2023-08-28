import torch
import torch.nn as nn

class CrossAttn(nn.Module):
    def __init__(self,  cfg):
        super(CrossAttn, self).__init__()
        
        # struct
        self.Linear1 = nn.Linear(cfg.CrossAttn.d_struct, cfg.CrossAttn.d_model)
        
        # seq
        self.Embedding = nn.Embedding(cfg.CrossAttn.n_vocab, cfg.CrossAttn.d_model)
        
        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.CrossAttn.d_model,
            num_heads=cfg.CrossAttn.n_head,
            dropout=cfg.CrossAttn.dropout)
        
        # Transformer
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.CrossAttn.d_model,
            nhead=cfg.CrossAttn.n_head,
            dim_feedforward=cfg.CrossAttn.d_ffn,
            dropout=cfg.CrossAttn.dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=cfg.CrossAttn.nlayer
        )
        
        self.Linear2 = nn.Linear(cfg.CrossAttn.d_model, cfg.CrossAttn.n_vocab)
        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        
    def forward(self, struct, seq, seq_mask):
        # 
        struct = self.Linear1(struct)
        seq = self.Embedding(seq)
        # cross attention forward
        cross_attn_output, _ = self.cross_attn(struct, seq, seq)
        # fused information
        seq = seq + cross_attn_output
        # Encoder 
        output = self.transformer_encoder.forward(seq,mask=seq_mask)
        output = self.Linear2(output)
        return output
    
    def compute_loss(self, struct, seq, seq_mask):
        seq_recovered = self.forward(struct, seq, seq_mask)
        #seq = seq.reshape(seq.shape[0] * seq.shape[1])
        seq = nn.functional.one_hot(seq,21)
        #loss = self.CrossEntropyLoss(seq_recovered, seq.to(torch.float32))
        loss = self.CrossEntropyLoss(seq_recovered.view(-1, seq_recovered.size(-1)), seq.view(-1))
        return loss
        