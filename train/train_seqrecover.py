from dataset.data import StructureDataset, StructureLoader
import torch
import torch.nn as nn
from model.s2s.s2s import S2S, SeqRecover
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf
from matplotlib import pyplot as plt
import numpy as np
import random

cfg = OmegaConf.load('./config/s2s.yaml')

writer = SummaryWriter('./logs/SeqRecovery')

losses = []

def train():
    model = SeqRecover(cfg=cfg)
    dataset = StructureDataset('./dataset/demo.jsonl')
    #random.shuffle(dataset)
    split_index = int(0.8 * len(dataset))
    train_dataset = dataset[:split_index]
    val_dataset = dataset[split_index:]
    
    train_dataloader = StructureLoader(train_dataset, batch_size=32)
    val_dataloader = StructureLoader(val_dataset, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.TRAIN.lr)
    
    for epoch in range(cfg.TRAIN.epochs):
        train_batch_losses = []
        for i,batch in enumerate(train_dataloader):
            # if i in list(range(302)):
            #     continue
            
            struct, seq, mask = batch.values()
            loss = model.compute_loss(
                struct=struct,
                seq=torch.zeros_like(seq),
                seq_real=seq,
                struct_mask=None,
                # struct_padding_mask=None,
                # seq_padding_mask=None
                struct_padding_mask=mask.bool(),
                seq_padding_mask=mask.bool()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.detach().numpy(),'batch shape is:', struct.size(),flush=True)
            losses.append(loss.detach().numpy())


if __name__ == '__main__':
    train() 
    plt.plot(np.array(range(len(losses))), losses)
    plt.show()
            
        