from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import torch
import tqdm

import numpy as np
import pickle as pkl
import xrecord

# define the class of dataloader
class Test(Dataset):
    def __init__(self):
        self.rec = xrecord.XRecord('test_data', 'r')
        self.keys = self.rec.keys
    
    def __getitem__(self, index):
        key = self.keys[index]
        data_byte = self.rec.read(key)
        pc = pkl.loads(data_byte)
        pc = torch.FloatTensor(pc)
        return pc
    
    def __len__(self):
        return len(self.keys)


test_dataset = Test()
test_loader = DataLoader(test_dataset,
                        batch_size=2,
                        shuffle=True,
                        num_workers=4,
                        pin_memory=True)

for data in tqdm.tqdm(test_loader):
    pass