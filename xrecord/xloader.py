import numpy as np
import os

import multiprocessing


class XDataset(object):
    """
    A dataset class for data process
    """
    def __init__(self):
        pass

    def read_iter(self):
        """
        iterative function
        """
        pass

    def __len__(self):
        pass


class XLoader(object):
    """
    A fast dataset loader with multi-process
    """
    def __init__(self, fname, batch_size=1, shuffle=False, total_sample_num=None, num_workers=None, rank=0, world_size=1, collate_fn=None):
        """
        fname: str or list of str
        batch_size: batch_size on each node
        shuffle: whether to shuffle the sample
        total_sample_num: define the total sample number, it will read from file, if total_sample_num=None
        num_workers: number of workers on each node
        rank: rank of the node
        world_size: total number of the nodes
        collate_fn: merges a list of samples to form a mini-batch of Tensor
        """
        self.fname = fname
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.total_sample_num = total_sample_num
        self.num_workers = num_workers
        self.rank = rank
        self.world_size = world_size
        self.collate_fn = collate_fn

        assert self.rank < self.world_size

        self.shuffle_queue_size = self.batch_size * 32

        # data processing utilities
        self.transform = transform
        self.transform_kwargs = transform_kwargs

        # save parameters as properties
        self.data_size = int(os.popen('wc -l {}.idx'.format(self.record_file)).read().split()[0])
        if self.total_sample is not None:
            self.data_size = self.total_sample

        self.shuffle = shuffle
        self.random_state = np.random.RandomState(seed=5)

        # partition data
        self.partition_count = (self.data_size + self.num_partition - 1) // self.num_partition
        self.partition_count = ((self.partition_count + batch_size - 1) // batch_size) * batch_size

        print("loader rank:{}, partition count:{}".format(self.rank, self.partition_count))

        # decide data and label names
        self.data_name = data_name
        self.label_name = label_name

        # status variable for counting
        self._cur = 0

        # multi-thread settings
        self.num_worker = num_worker

        if (worker_queue_depth is None):
            worker_queue_depth = self.batch_size * len(ctx) * 10
        
        self.data_queue = multiprocessing.Queue(maxsize=worker_queue_depth)

        # get first batch to fill in provide_data and provide_label
        self.reset()
        self._thread_start()

    @property
    def total_record(self):
        return self.partition_count

    def _thread_start(self):
        # load data thread
        workers = [multiprocessing.Process(target=self.worker, args=[i]) for i in range(self.num_worker)]
        for w in workers:
            w.daemon = True
            w.start()

    def reset(self):
        # reset count
        self._cur = 0

    def iter_next(self):
        return self._cur + self.batch_size <= self.total_record

    def __iter__(self):
        return self

    def __next__(self):
        if self.iter_next():
            self._cur += self.batch_size
            result = self.load_batch()
            return result
        else:
            raise StopIteration

    def transform_iter(self, it, context):
        for data in it:
            for meta_data in self.transform(data, self.transform_kwargs, context):
                self.data_queue.put(meta_data)

    def transform_iter_shuffle(self, it, context):
        buffers = []
        for data in it:
            for meta_data in self.transform(data, self.transform_kwargs, context):
                if len(buffers) >= self.shuffle_queue_size:
                    random_index = np.random.randint(len(buffers))
                    item_tmp = buffers[random_index]
                    buffers[random_index] = meta_data
                    self.data_queue.put(item_tmp)
                else:
                    buffers.append(meta_data)
        
        self.random_state.shuffle(buffers)
        for meta_data in buffers:
            self.data_queue.put(meta_data)

    def worker(self, idx):
        while True:
            description = {"data": "byte"}
            shard = (idx * self.num_partition + self.rank, self.num_worker * self.num_partition)
            context = self.ctx[idx % len(self.ctx)]
            it = reader.tfrecord_loader('{}.rec'.format(self.record_file), '{}.idx'.format(self.record_file), description, shard)
            if self.shuffle:
                self.transform_iter_shuffle(it, context)
            else:
                self.transform_iter(it, context)

            # transfer into next data block
            self.rank = self.rank + 1
            self.rank = self.rank % self.num_partition

    def load_batch(self):
        records = []
        for i in range(self.batch_size):
            records.append(self.data_queue.get())

        data = []
        provide_data = []
        for name in self.data_name:
            mx_data = mx.nd.array(np.stack([r[name] for r in records]))
            data.append(mx_data)
            provide_data.append((name, mx_data.shape))

        label = []
        provide_label = []
        for name in self.label_name:
            mx_data = mx.nd.array(np.stack([r[name] for r in records]))
            label.append(mx_data)
            provide_label.append((name, mx_data.shape))

        file_name_list = [r['file_name'] for r in records]
        data_batch = mx.io.DataBatch(data=data,
                                    label=label,
                                    provide_data=provide_data,
                                    provide_label=provide_label)
        
        return (data_batch, file_name_list)

    def __len__(self):
        return self.total_record