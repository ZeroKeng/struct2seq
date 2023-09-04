from model.mpnn.mpnn import EncLayer, DecLayer,ProteinMPNN
import torch
import torch.nn as nn
from dataset.data import StructureDataset, StructureLoader
import random


# E_idx = torch.zeros(32,100,30).to(torch.long)
# mask = torch.zeros(32,100)
# X = torch.zeros(32,100,30,4)
# S = torch.zeros(32,100)
mpnn = ProteinMPNN(edge_features=4, hidden_dim=8,num_encoder_layers=1, num_decoder_layers=1,dropout=0)
# a = mpnn(X,S.to(torch.long),mask,E_idx)
# print(a.shape)


sd = StructureDataset('dataset/demo.jsonl')
sl = StructureLoader(sd, batch_size=128)


loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mpnn.parameters(),lr=0.001)

for epoch in range(100):
    for batch in sl:
        X, S, mask, indices = batch.values()
        #print(X.shape, S.shape, mask.shape, indices.shape)
        indices = indices.to(torch.long)
        S_zero = torch.zeros_like(S)
        S = torch.nn.functional.one_hot(S,21).to(torch.float32)
        
        
        optimizer.zero_grad()
        output = mpnn(X, S_zero, mask, indices)
        loss = loss_fn(output,S)
        loss.backward()
        optimizer.step()
        print(epoch, loss)
        
        
    
    
