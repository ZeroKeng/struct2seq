from numpy import cross, shape
from model.net.crossattn import CrossAttn
import torch
from torch import nn, Tensor
from torch.nn import functional as F 
from omegaconf import DictConfig, OmegaConf
from dataset.data import StructureDataset, StructureLoader
from matplotlib import pyplot as plt
import numpy as np

cfg = OmegaConf.load('./config/s2s.yaml')

struct = torch.randn((32,100,120))
seq = torch.randn((32,100)).to(torch.long)


dataset = StructureDataset('./dataset/demo.jsonl')
dataloader = StructureLoader(dataset=dataset,batch_size=32)

crossattn = CrossAttn(cfg = cfg)

optimizer = torch.optim.Adam(crossattn.parameters(), lr=cfg.TRAIN.lr)

# for epoch in range(10):
#     for batch in dataloader:
#         struct, seq, mask = batch.values()
#         print(struct.shape, seq.shape)
#         loss = crossattn.compute_loss(struct,seq,None)
#         print(loss)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         print(epoch, loss.detach().numpy(),flush=True)

losses = []

for epoch in range(20):
    for batch in dataloader:
        struct, seq, mask = batch.values()
        print(seq)
        batch_length = struct.shape[1]
        loss = crossattn.compute_loss(struct, seq, None)
        losses.append(loss.detach().numpy() / batch_length)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.detach().numpy(), batch_length, loss.detach().numpy() / batch_length)
        

plt.plot(np.array(range(len(losses))), losses)
plt.show()