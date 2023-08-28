import torch
import torch.nn as nn
from model.s2s.s2s import S2S
from dataset.data import StructureDataset, StructureLoader
from omegaconf import DictConfig, OmegaConf

cfg = OmegaConf.load('./config/s2s.yaml')
cfg = cfg.S2S


dataset = StructureDataset('./dataset/demo.jsonl')
dataloader = StructureLoader(dataset=dataset,batch_size=32)
model = S2S(cfg=cfg)

for batch in dataloader:
    struct, seq, mask = batch.values()
    print(mask.shape)
    model(struct,seq,None,None,mask,mask)