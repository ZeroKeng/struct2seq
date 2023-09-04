from turtle import forward
import torch
import torch.nn as nn

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        
    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """
        h_V: [B,N,C]
        h_E: [B,N,K,C]
        E_idx: [B,N,K]
        """
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) # h_V is [B,N,C], E_idx is [B,N,K] output is [B,N,K,C], h_E is [B,N,K,C], h_EV is [B,N,K,2C]
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1) # turn h_V [B,N,C] into h_V_expand [B,N,K,C] 
        h_EV = torch.cat([h_V_expand, h_EV], -1) # [B,N,K,3C]
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV))))) #[B,N,K,C]
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale # [B,N,C]
        h_V = self.norm1(h_V + self.dropout1(dh)) # [B,N,C]
        dh = self.dense(h_V) #[B,N,C]
        h_V = self.norm2(h_V + self.dropout2(dh)) #[B,N,C]
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx) #[B,N,K,2C] 
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1) #[B,N,K,C]
        h_EV = torch.cat([h_V_expand, h_EV], -1) #[B,N,K,3C]
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV))))) # [B,N,K,C]
        h_E = self.norm3(h_E + self.dropout3(h_message)) # [B,N,K,C]
        return h_V, h_E # [B,N,C] [B,N,K,C]
    
class DecLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout = 0.1, num_heads=None, scale=30):
        super().__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True) # 4C -> C
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)
        
    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1) # [B,N,K,C]
        h_EV = torch.cat([h_V_expand, h_E], -1) # [B, N, K, 4C], so h_E is 3C

        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV))))) # [B,N,K,C]
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale #[B,N,C]

        h_V = self.norm1(h_V + self.dropout1(dh)) # [B,N,C]

        # Position-wise feedforward
        dh = self.dense(h_V) # [B,N,C]
        h_V = self.norm2(h_V + self.dropout2(dh)) # [B,N,C]

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V #[B,N,C]

class ProteinMPNN(nn.Module):
    def __init__(self,num_letters=21, node_features=128, edge_features=120,
                 hidden_dim = 128, num_encoder_layers=3,num_decoder_layers=3,
                 vocab=21, k_neighbor=32, dropout=0.1):
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            EncLayer(hidden_dim, hidden_dim*2, dropout=dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            DecLayer(hidden_dim, hidden_dim*3, dropout=dropout)
            for _ in range(num_decoder_layers)
        ])
        self.W_out = nn.Linear(hidden_dim, num_letters, bias=True)
        
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, X, S, mask, E_idx):
        E = X
        h_E = self.W_e(E) #[B,N,K,C]
        h_V = torch.zeros((E.shape[0], E.shape[1], h_E.shape[-1])) #[B,N,C]
        mask_attend = gather_nodes(mask.unsqueeze(-1),  E_idx).squeeze(-1) #gather_nodes([B,N,C], [B,N,K]) = [B,N,K,C] -> [B,N,K]
        mask_attend = mask.unsqueeze(-1) * mask_attend #[B,N,1] * [B,N,K] = [B,N,K]
        for layer in self.encoder_layers:
            h_V, h_E = layer(h_V, h_E, E_idx, mask, mask_attend)
        h_S = self.W_s(S) #[B,N,C]
        h_ES = cat_neighbors_nodes(h_S, h_E, E_idx) #[B,N,,2C]
        h_EX_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_E, E_idx) #[B,N,K,2C]
        h_EXV_encoder = cat_neighbors_nodes(h_V, h_EX_encoder, E_idx) #[B,N,K,3C]
        chain_M=mask.clone()
        decoding_order = torch.argsort((chain_M+0.0001)*(torch.abs(torch.randn(chain_M.shape))))
        mask_size = E_idx.shape[1]
        permutation_matrix_reverse = torch.nn.functional.one_hot(decoding_order, num_classes=mask_size).float()
        order_mask_backward = torch.einsum('ij, biq, bjp->bqp',(1-torch.triu(torch.ones(mask_size,mask_size))), permutation_matrix_reverse, permutation_matrix_reverse)
        mask_attend = torch.gather(order_mask_backward, 2, E_idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        
        h_EXV_encoder_fw = mask_fw * h_EXV_encoder
        for layer in self.decoder_layers:
            h_ESV = cat_neighbors_nodes(h_V, h_ES, E_idx) #[B,N,K,3C]
            h_ESV = mask_bw * h_ESV + h_EXV_encoder_fw
            h_V = layer(h_V, h_ESV, mask)
        
        logits = self.W_out(h_V)
        #log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        logits = torch.nn.functional.softmax(logits,dim=-1)
        return logits
        
        

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h
    
    
def gather_nodes(nodes, neighbor_idx):
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1)) # [B,N,K] -> [B, N*K]
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2)) # [B,N*K,1] -> [B,N*K,C]
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat) # [B,NK,C]
    #neighbor_features = torch.gather(nodes, 1, neighbors_flat.unsqueeze(-1).expand(-1, -1, -1, nodes.size(2))) # 
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1]) #[B,N,K,C]
    return neighbor_features

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx) #[B,N,K,C]
    h_nn = torch.cat([h_neighbors, h_nodes], -1) # torch.cat([B,N,K,C],[B,N,K,C])
    return h_nn #[B,N,K,2C]