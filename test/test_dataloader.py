from dataset.data import StructureLoader
from dataset.data import StructureDataset



if __name__ == '__main__':
    
    sd = StructureDataset('./dataset/demo.jsonl')
    sl = StructureLoader(dataset=sd,batch_size=32)
    
    a = next(iter(sl))
    print(a['struct'].shape)
    print(a['seq'].shape)
    print(a['mask'].shape)
    
    