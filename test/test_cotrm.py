from model.net.cotrm import CoTRM
import torch
from torch import nn, Tensor
from torch.nn import functional as F 
from omegaconf import DictConfig, OmegaConf

cfg = OmegaConf.load('./config/s2s.yaml')
cfg = cfg.S2S

struct = torch.randn((32,100,512))
seq = torch.randn((32,100,512))

cotrm = CoTRM(cfg = cfg)

cotrm.forward(struct,seq,None,None,None,None)