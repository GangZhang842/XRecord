import numpy as np

import torch
import random
import os

from abc import abstractmethod, ABCMeta

import multiprocessing


class XDataset(metaclass=ABCMeta):
    """
    A dataset class for data process
    """
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def gene_iter(self, part=None):
        """
        iterative function for generating data.
        part=(part_id, part_num) means the total dataset is split into "part_num" parts
        and the $part_id_{th}$ part is used, which is required in the distributed training system.
        By default, part is set as None, meaning that the whole dataset is loaded.
        """
        pass
    
    @abstractmethod
    def __len__(self):
        pass


def collate_fn_meta(meta_data):
    if isinstance(meta_data[0], np.ndarray):
        data_out = torch.as_tensor(np.stack(meta_data, axis=0))
        return data_out
    elif isinstance(meta_data[0], torch.Tensor):
        data_out = torch.stack(meta_data, dim=0)
        return data_out
    else:
        return meta_data


def collate_fn_default(batch):
    result = []
    if type(batch[0]) in [list, tuple]:
        for batch_meta_data in zip(*batch):
            result.append(collate_fn_meta(batch_meta_data))
        return result
    else:
        return collate_fn_meta(batch)


class XTrainLoader(object):
    """
    A fast dataset loader for distributed training
    """
    def __init__(self, xdataset_class, xdataset_args, batch_size=1, num_workers=1, rank=0, world_size=1, collate_fn=collate_fn_default):
        """
        xdataset_class: a subclass of XDataset
        xdataset_args: arguments of the subclass
        batch_size: batch_size on each node
        num_workers: number of workers on each node
        rank: rank of the node
        world_size: total number of the nodes
        collate_fn: merges a list of samples to form a mini-batch of Tensor
        """
        self.xdataset_class = xdataset_class
        self.xdataset_args = xdataset_args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.collate_fn = collate_fn

        assert issubclass(self.xdataset_class, XDataset)
        assert type(self.xdataset_args) in [list, tuple]
        assert self.num_workers >= 1
        assert self.rank < self.world_size

        # shuffle data parameters
        self.shuffle_queue_size = self.batch_size * 64

        # partition data
        self.xdataset_obj_list = [self.xdataset_class(*self.xdataset_args) for i in range(self.num_workers)]
        self.partition_count = (len(self.xdataset_obj_list[0]) + self.world_size - 1) // self.world_size
        self.partition_count = (self.partition_count + self.batch_size - 1) // self.batch_size

        print("XLoader rank:{}, partition count:{}".format(self.rank, self.partition_count))

        # multi-process settings
        worker_queue_depth = self.batch_size * 64
        self.worker_queue = multiprocessing.Queue(maxsize=worker_queue_depth)

        # status variable for counting
        self.__cur = 0

        # start multi-process
        self.__reset()

        # start worker
        self.workers = [multiprocessing.Process(target=self.__worker, args=[i, self.xdataset_obj_list[i]]) for i in range(self.num_workers)]
        for w in self.workers:
            w.daemon = True
            w.start()
    
    def __del__(self):
        self.__shutdown()
    
    def __shutdown(self):
        if len(self.workers) > 0:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
                while w.is_alive():
                    pass
                w.close()
    
    def __worker(self, idx, xdataset_obj):
        rank = self.rank
        while True:
            buffers = []
            part = (idx * self.world_size + rank, self.num_workers * self.world_size)
            for data in xdataset_obj.gene_iter(part):
                if len(buffers) >= self.shuffle_queue_size:
                    random_index = np.random.randint(len(buffers))
                    item_tmp = buffers[random_index]
                    buffers[random_index] = data
                    self.worker_queue.put(item_tmp)
                else:
                    buffers.append(data)
            
            random.shuffle(buffers)
            for data in buffers:
                self.worker_queue.put(data)
            
            # transfer into next data block
            rank = rank + 1
            rank = rank % self.world_size
    
    def __reset(self):
        # reset count
        self.__cur = 0
    
    def __iter__(self):
        self.__reset()
        return self
    
    def __next__(self):
        if self.__cur < len(self):
            result = self.collector()
            self.__cur += 1
            return result
        else:
            raise StopIteration
    
    def collector(self):
        records = []
        for i in range(self.batch_size):
            records.append(self.worker_queue.get())
        
        if self.collate_fn is None:
            return records
        else:
            batch_data = self.collate_fn(records)
            return batch_data
    
    def __len__(self):
        return self.partition_count


class XTestLoader(object):
    """
    A fast dataset loader for distributed testing
    """
    def __init__(self, xdataset_class, xdataset_args, batch_size=1, num_workers=1, rank=0, world_size=1, collate_fn=collate_fn_default):
        """
        xdataset_class: a subclass of XDataset
        xdataset_args: arguments of the subclass
        batch_size: batch_size on each node
        num_workers: number of workers on each node
        rank: rank of the node
        world_size: total number of the nodes
        collate_fn: merges a list of samples to form a mini-batch of Tensor
        """
        self.xdataset_class = xdataset_class
        self.xdataset_args = xdataset_args
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.collate_fn = collate_fn

        assert issubclass(self.xdataset_class, XDataset)
        assert type(self.xdataset_args) in [list, tuple]
        assert self.num_workers >= 1
        assert self.rank < self.world_size

        # multi-process settings
        worker_queue_depth = self.batch_size * 64
        self.worker_queue = multiprocessing.Queue(maxsize=worker_queue_depth)

        self.xdataset_obj_list = [self.xdataset_class(*self.xdataset_args) for i in range(self.num_workers)]

        self.workers = []
    
    def __del__(self):
        self.__shutdown()
    
    def __shutdown(self):
        if len(self.workers) > 0:
            for w in self.workers:
                if w.is_alive():
                    w.terminate()
                while w.is_alive():
                    pass
                w.close()
    
    def __worker(self, idx, xdataset_obj):
        part = (idx * self.world_size + self.rank, self.num_workers * self.world_size)
        for data in xdataset_obj.gene_iter(part):
            self.worker_queue.put(data)
    
    def __reset(self):
        # clear existing resouces
        self.__shutdown()
        while not self.worker_queue.empty():
            self.worker_queue.get()
        
        # re-allocate resouces
        self.workers = [multiprocessing.Process(target=self.__worker, args=[i, self.xdataset_obj_list[i]]) for i in range(self.num_workers)]
        for w in self.workers:
            w.daemon = True
            w.start()
    
    def __iter__(self):
        self.__reset()
        return self
    
    def __is_all_done(self):
        for w in self.workers:
            if w.is_alive():
                return False
        return True
    
    def __next__(self):
        result = self.collector()
        if result is not None:
            return result
        else:
            raise StopIteration
    
    def collector(self):
        records = []
        if self.__is_all_done() and self.worker_queue.empty():
            return None
        else:
            for i in range(self.batch_size):
                try:
                    records.append(self.worker_queue.get(timeout=2))
                except:
                    pass
            
            length_tmp = len(records)
            if length_tmp == 0:
                return None
            for i in range(length_tmp, self.batch_size):
                records.append(records[0])
        
        if self.collate_fn is None:
            return records
        else:
            batch_data = self.collate_fn(records)
            return batch_data
    
    def __len__(self):
        raise NotImplementedError