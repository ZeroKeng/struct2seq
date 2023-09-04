from model.net.cotrm import CoTRM
import torch
from torch import nn, Tensor
from torch.nn import functional as F 
from omegaconf import DictConfig, OmegaConf
from dataset.data import StructureDataset, StructureLoader

cfg = OmegaConf.load('./config/s2s.yaml')
cfg = cfg.S2S

dataset = StructureDataset('./dataset/demo.jsonl')
dataloader = StructureLoader(dataset=dataset,batch_size=32)

cotrm = CoTRM(cfg)