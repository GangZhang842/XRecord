import numpy as np
import pickle as pkl
import xrecord

import gzip
import os

import tqdm
import time


DataDir = './'

rec = xrecord.XRecordReadList([os.path.join(DataDir, 'test_data_{}'.format(i)) for i in range(3)])


for i in [2, 3, 4, 0]:
    for data_byte in rec.read_iter((i, 7)):
        data = pkl.loads(data_byte)
        print(data.max(), data.min())