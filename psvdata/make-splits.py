from psv.ptdataset import PsvDataset
from random import shuffle
from psv.config import C
import os

ds = PsvDataset()

samples = [os.path.relpath(xml, start=C.DATA_ROOT) for  xml in ds.xmls]
shuffle(samples)

num_train = int(len(samples)*0.80)
train = samples[:num_train]
val = samples[num_train:]

with open(os.path.join(C.DATA_ROOT, 'train.txt'), 'w') as f:
    f.write('\n'.join(train))

with open(os.path.join(C.DATA_ROOT, 'val.txt'), 'w') as f:
    f.writelines('\n'.join(val))

