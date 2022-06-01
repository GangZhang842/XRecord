import numpy as np
import pickle as pkl
import xrecord

import gzip
import os

import tqdm
import time


DataDir = './'

rec = xrecord.XRecord(os.path.join(DataDir, 'test_data_{}'.format(0)))

for data_byte in rec.read_iter((0, 2)):
    data = pkl.loads(data_byte)
    print(data.max(), data.min())