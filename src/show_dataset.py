import argparse
import io
import cPickle as pickle
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(description='script to show training data as image')
parser.add_argument('input', type=str, help='input dataset file path')
parser.add_argument('output', type=str, help='output image file path')
parser.add_argument('--width', '-W', default=96, type=int, help='character data width')
parser.add_argument('--height', '-H', default=96, type=int, help='character data height')
args = parser.parse_args()

with open(args.input, 'rb') as f:
    images, labels = pickle.load(f)

length = len(images)

print 'data length: {}'.format(length)

w = args.width
h = args.height
row_num = 10
col_num = 10
image_num = row_num * col_num
out_images = np.zeros((row_num, col_num, h, w))
for i, image in enumerate(images[:image_num / 2] + images[-image_num / 2:]):
    with io.BytesIO(image) as binary:
        out_images[i / col_num, i % col_num,:,:] = np.asarray(Image.open(binary).convert('L').resize((w, h))).reshape(h, w)
Image.fromarray(255 - out_images.astype(np.uint8).transpose(0, 2, 1, 3).reshape((h * row_num, w * col_num))).save(args.output)
