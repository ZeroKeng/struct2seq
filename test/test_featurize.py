from dataset.data import StructureDataset, StructureLoader
from dataset.featurize import protein_featurize
import torch
import torch.nn as nn

sd = StructureDataset('./dataset/demo2.jsonl')
sl = StructureLoader(sd, batch_size=32)

for batch in sl:
    X, S, mask = batch.values()
    print(S[:,0])