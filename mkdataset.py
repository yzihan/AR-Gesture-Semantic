#!/usr/bin/env python

import cv2
import os
from functools import reduce
import random
import pickle

train_videos = [{'fn': 'a.mp4', 'tag': 'Closed'}, {'fn': 'b.mp4', 'tag': 'Open'}]
test_videos = [{'fn': 'c.mp4'}]

def processor(v):
    v['cap'] = cv2.VideoCapture(v['fn'])
    v['open'] = v['cap'].isOpened()
    v['data'] = []
    retry = 5
    while True:
        ok, f = v['cap'].read()
        if not ok:
            if retry > 0:
                retry -= 1
                continue
            break
        v['data'].append(f)
    return v

def getdata(p, v):
    p.extend(map(lambda vi: {'frame': vi, 'tag': v['tag'] if 'tag' in v else v['fn']}, v['data']))
    return p

def mkdir(p):
    if not os.path.isdir(p):
        os.mkdir(p)

def extract(videos, fn, name, shuffle=True):
    extracted = list(map(processor, videos))

    for p in extracted:
        print('{}: {}: {} frame(s)'.format(name, p['fn'],len(p['data'])))

    data = reduce(getdata, extracted, [])
    if shuffle:
        random.shuffle(data)

    index = []
    i = 0
    for k in range(0, len(data), 128):
        part = data[k : k+128]
        i += 1
        part_fn = fn + '.part{}.pickle'.format(i)
        index.append({'fn': part_fn, 'len': len(part)})
        with open(part_fn, 'wb') as f:
            pickle.dump(part, f)
    with open(fn + '.index.pickle', 'wb') as f:
        pickle.dump(index, f)

extract(train_videos, 'trainset', 'train')
extract(test_videos, 'testset', 'test', False)
