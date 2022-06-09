from xrecord import XDataset, XTrainLoader, XTestLoader

class TrainDataset(XDataset):
    def __init__(self, num):
        self.num = num
    
    def gene_iter(self, part=None):
        part_id, part_num = part
        start_idx = self.num * part_id // part_num
        end_idx = self.num * (part_id + 1) // part_num
        for i in range(start_idx, end_idx):
            yield i
    
    def __len__(self):
        return self.num


xdataset = TrainDataset(20)
test_loader = XTestLoader(xdataset, batch_size=6, num_workers=3, rank=0, world_size=1)

for e in range(5):
    for i in test_loader:
        print(i)