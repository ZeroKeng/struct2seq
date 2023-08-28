from dataset.data import StructureDataset, StructureLoader
import torch
import torch.nn as nn
from model.s2s.s2s import S2S, SeqRecover, StructRecover
from torch.utils.tensorboard import SummaryWriter
from omegaconf import DictConfig, OmegaConf

cfg = OmegaConf.load('./config/s2s.yaml')

writer = SummaryWriter('./logs/SeqRecovery')

def train():
    model = StructRecover(cfg=cfg)
    dataset = StructureDataset('./dataset/chain_set.jsonl')
    dataloader = StructureLoader(dataset, batch_size=32)
    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.TRAIN.lr)
    
    for epoch in range(cfg.TRAIN.epochs):
        train_batch_losses = []
        for i,batch in enumerate(dataloader):
            if i in (10,20):
                continue
            struct, seq, mask = batch.values()
            loss = model.compute_loss(
                struct=struct,
                seq=seq,
                seq_mask=None,
                struct_padding_mask=mask.bool(),
                seq_padding_mask=mask.bool()
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(epoch, loss.detach().numpy(),flush=True)

if __name__ == '__main__':
    train() 