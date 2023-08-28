from sys import get_coroutine_origin_tracking_depth
from dataset.data import StructureLoader
from dataset.data import StructureDataset
from dataset.featurize import get_coords6d

if __name__ == '__main__':
    
    sd = StructureDataset('./dataset/demo.jsonl')
    sl = StructureLoader(dataset=sd,batch_size=32)
    
    for batch in sl:
        X = batch['struct']
        for i in X:
            sixd = get_coords6d(i.numpy())
            print(sixd.shape)