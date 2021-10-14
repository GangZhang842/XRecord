## XRecord
**XRecord** is a general key-value dataset for fast data access.

#### 1 Dependency
```bash
Python>=3.5
```
#### 2 Quick start
##### 2.1 Installation
```bash
python3 setup.py install
```
##### 2.2 Write XRecord
```bash
from xrecord import XRecord
import pickle as pkl

rec = XRecord('data', mode='w') #write
rec = XRecord('data', mode='a') #append

key = '0'
data_byte = pkl.dumps(data) #serialize
rec.write(key, data_byte)

rec.close() #close file
```
Note that **XRecord** does not support multi-process write!
##### 2.3 Read XRecord
```bash
from xrecord import XRecord
import pickle as pkl

rec = XRecord('data', mode='r') #read
# key-value
for key in rec.keys:
  data_byte = rec.read(key)
  data = pkl.load(data_byte)

# iterator
for data_byte in rec:
  data = pkl.load(data_byte)
rec.close()
```
