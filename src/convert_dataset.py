import numpy as np
import io
import os
import sys
from itertools import chain
import cPickle as pickle
from PIL import Image
from PIL import ImageEnhance
import StringIO
import binary

def _flatten(l):
    return list(chain.from_iterable(l))

DATASET_FILE_NUM = 50
RECORD_NUM = 12144
RECORD_SIZE = 8199
IN_IMAGE_WIDTH = 128
IN_IMAGE_HEIGHT = 127
IMAGE_SIZE = IN_IMAGE_HEIGHT * IN_IMAGE_WIDTH
OUT_IMAGE_WIDTH = 96
OUT_IMAGE_HEIGHT = 96
CODES = _flatten([
    [ 0x2422, 0x2424, 0x2426, 0x2428 ],
    range(0x242a, 0x2443),
    range(0x2444, 0x2463),
    [ 0x2464, 0x2466 ],
    range(0x2468, 0x246e),
    [ 0x246f, 0x2472, 0x2473],
    _flatten(map(lambda x: range(x + 0x21, x + 0x7f), range(0x3000, 0x4f00, 0x100))),
    range(0x4f21, 0x4f54),
])
assert len(CODES) == 3036
CODES_TO_INDEX = dict(zip(CODES, range(len(CODES))))

def parse_record(record):
    code = binary.read16be(record, 2)
    data = np.asarray(binary.read8s(record, IMAGE_SIZE / 2, 64), dtype=np.uint8).reshape((IMAGE_SIZE / 2, 1))
    x = data >> 4
    y = data & 0xf
    image = np.concatenate((x, y), axis=1).reshape(IMAGE_SIZE)
    return (CODES_TO_INDEX[code], image)

def read_file(path):
    labels = np.zeros((RECORD_NUM,), np.uint16)
    images = np.zeros((RECORD_NUM, IMAGE_SIZE), np.uint8)
    with open(path, 'rb') as f:
        for i in range(RECORD_NUM):
            labels[i], images[i] = parse_record(f.read(RECORD_SIZE))
    return (images, labels)

if __name__=='__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    lagels_list = []
    image_list = []
    index = 0

    for i in range(DATASET_FILE_NUM):
        path = os.path.join(in_dir, 'ETL9G_{0:02d}'.format(i + 1))
        print 'converting {}...'.format(path)
        images, labels = read_file(path)
        for image in images:
            with io.BytesIO() as b:
                image_instance = Image.fromarray(image.reshape((IN_IMAGE_HEIGHT, IN_IMAGE_WIDTH)) * 17).resize((OUT_IMAGE_HEIGHT, OUT_IMAGE_WIDTH))
                # original image contrast is too weak
                ImageEnhance.Contrast(image_instance).enhance(3).save(b, format='png')
                image_list.append(b.getvalue())
        lagels_list.append(labels)
    path = os.path.join(out_dir, 'etl9g.pkl')
    with open(path, 'wb') as f:
        pickle.dump((image_list, np.concatenate(lagels_list)), f, pickle.HIGHEST_PROTOCOL)
    print 'done'
