import numpy as np
import os
import sys
from itertools import chain
import cPickle as pickle
import binary

def flatten(l):
    return list(chain.from_iterable(l))

RECORD_NUM = 12144
RECORD_SIZE = 8199
CODES = flatten([
    [ 0x2422, 0x2424, 0x2426, 0x2428 ],
    range(0x242a, 0x2443),
    range(0x2444, 0x2463),
    [ 0x2464, 0x2466 ],
    range(0x2468, 0x246e),
    range(0x246f, 0x2474),
    flatten(map(lambda x: range(x + 0x21, x + 0x7f), range(0x3000, 0x4f00, 0x100))),
    range(0x4f21, 0x4f54),
    flatten(map(lambda x: range(x + 0x21, x + 0x7f), range(0x5000, 0x7400, 0x100))),
    range(0x7421, 0x7427),
])
CODES_TO_INDEX = dict(zip(CODES, range(len(CODES))))

def parse_record(record):
    code = binary.read16be(record, 2)
    data = np.asarray(binary.read8s(record, 8128, 64), dtype=np.uint8).reshape((8128, 1))
    x = data >> 4
    y = data & 0xf
    image = np.concatenate((x, y), axis=1).reshape(127 * 128)
    return (CODES_TO_INDEX[code], image)

def read_file(path):
    labels = np.zeros((RECORD_NUM,), np.uint16)
    images = np.zeros((RECORD_NUM, 127 * 128), np.uint8)
    with open(path, 'rb') as f:
        for i in range(RECORD_NUM):
            labels[i], images[i] = parse_record(f.read(RECORD_SIZE))
    return (images, labels)

in_dir = sys.argv[1]
out_dir = sys.argv[2]

for i in range(50):
    path = os.path.join(in_dir, 'ETL9G_{0:02d}'.format(i + 1))
    print 'converting {}...'.format(path)
    with open(os.path.join(out_dir, 'etl9_{0:02d}.pkl'.format(i)), 'wb') as f:
        pickle.dump(read_file(path), f, pickle.HIGHEST_PROTOCOL)
    print 'done'
