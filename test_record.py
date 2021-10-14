import numpy as np
import pickle as pkl
import xrecord

import tqdm
import time
'''
rec = xrecord.XRecord('test_data', 'w')
for i in range(1000):
    print(i)
    key = str(i)
    data = np.zeros((200000, 32)).astype(np.float32) + i
    rec.write(key, pkl.dumps(data))

rec.close()
'''

rec = xrecord.XRecord('test_data', 'r')

key_list = [str(i) for i in range(1000)]
np.random.shuffle(key_list)

start = time.time()
for data_byte in tqdm.tqdm(rec):
    data = pkl.loads(data_byte)
    #error = np.abs(data - i)
    #print(data.shape, error.max())

end = time.time()
print("Cost: {}s".format(end - start))
rec.close()