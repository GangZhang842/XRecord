import numpy as np
import pickle as pkl
import xrecord

import gzip
import os

import tqdm
import time


DataDir = './'

rec = xrecord.XRecord(os.path.join(DataDir, 'test_data_{}'.format(2)), 'w')
for i in range(130, 160):
    print(i)
    key = str(i)
    data = np.zeros((200000, 32)).astype(np.float32) + i
    rec.write(key, pkl.dumps(data))

rec.close()