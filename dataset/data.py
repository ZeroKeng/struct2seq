import torch
import numpy as np
import json
from dataset.featurize import protein_featurize

class StructureDataset():
    """
    This is a class for sequence dataset
        1.read in a json file
        2.generate a iterator for coordinate of x,y,z of each atom
    """
    def __init__(self,
                 json_file,
                 alphabet='ACDEFGHIKLMNPQRSTVWY'):
        
        alphabet_set = set([aa for aa in alphabet])
        self.discard_count = {
            'bad_chars':0,
            'too_long':0
        }
        
        with open(json_file,'r') as f:
            self.data = []
            lines = f.readlines()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                coords = entry['coords']
                
                #convert coords in json to arrary in np
                for atom, coords in entry['coords'].items():
                    entry['coords'][atom] = np.asarray(coords)
                    
                #check for bad chars
                bad_char= set([aa for aa in seq]).difference(alphabet_set)
                if len(bad_char) == 0:
                    self.data.append(entry)
                else:
                    self.discard_count['bad_chars'] += 1
                    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self,idx):
        return self.data[idx]

class SequenceDataset():
    """
    This class create a sequence dataset from a json list file
        the same as the above SequenceDataset class
    """
    def __init__(self,jsonl_file,
                 alphabet='ACDEFGHIKLMNPQRSTVWY'):
        alphabet_set = set([aa for aa in alphabet])
        self.discard_count = {
            'bad_chars':0,
            'too_long':0
        }
        
        with open(jsonl_file,'r') as f:
            self.data = []
            lines = f.readlines()
            for i, line in enumerate(lines):
                entry = json.loads(line)
                seq = entry['seq']
                #check for bad chars
                bad_char = set([aa for aa in seq]).difference(alphabet_set)
                if len(bad_char) == 0:
                    self.data.append(seq)
                else:
                    self.discard_count['bad_chars'] += 1
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        return self.data[idx]


class SequenceLoader():
    """
    This class is used to load and batch a sequencedataset
        1. go through the dataset and
            1.1. sort the entries based on sequence length,
            1.2. group the entries based on batch size
    """
    def __init__(self, dataset, batch_size = 30):
        self.data = dataset
        self.batch_size = batch_size
        self.size = len(self.data)
        self.lengths = [len(self.data[i]) for i in range(self.size)]
        #sort the data
        sort_idx = np.argsort(self.lengths)
        batches = []
        batch = []
        for i in sort_idx:
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
                batch.append(self.data[i])
            else:
                batch.append(self.data[i])
            
        if len(batch) > 0:
            batches.append(batch)
            
        self.batches = batches
            
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch



class StructureLoader():
    """
    This class is to load protein structure and the mask information to show the missing value
    """
    def __init__(self,dataset,batch_size):
        self.data = dataset
        self.batch_size = batch_size
        self.lengths = [len(self.data[i]['seq']) for i in range(len(self.data))]
        sort_idx = np.argsort(self.lengths)
        batches = []
        batch = []
        for i in sort_idx:
            if len(batch) == self.batch_size:
                batches.append(batch)
                batch = []
                batch.append(self.data[i])
            else:
                batch.append(self.data[i])
        
        batches.append(batch)
        self.batches = batches
        #print(self.lengths, sort_idx)
    def __len__(self):
        return len(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            X, S, mask, indices = protein_featurize(batch)
            X[torch.isnan(X)] = 0.
            yield {'struct':X, 'seq':S, 'mask':mask, 'indices':indices}


# strucutre_dataset = StructureDataset('demo.jsonl')
# structure_loader = StructureLoader(strucutre_dataset,batch_size=30)

# batches = [b for b in structure_loader]
        

# print(batches[1][1])      
# sequence_dataset = SequenceDataset('demo.jsonl')
# print(len(sequence_dataset))
# print(sequence_dataset[1])
# print(sequence_dataset.discard_count)

# sequence_loader = SequenceLoader(sequence_dataset)

# for batch in sequence_loader:
#     print(batch)