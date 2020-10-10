#!/usr/bin/env python

import cv2
import os
from functools import reduce
import random
import pickle

# Train video and test video
# Frames are extracted and tagged to build the dataset
# the dict in the list is called "video structure" in the following comments, mostly use "v" as the variable
train_videos = [{'fn': 'a.mp4', 'tag': 'Closed'}, {'fn': 'b.mp4', 'tag': 'Open'}]
test_videos = [{'fn': 'c.mp4'}]

def processor(v):
    """
    create 'data' key in video structure, which is the list of all frames

    v: video structure (i.e. the dict inside train_videos and test_video list)

    usage: "map(processor, videos)"
    """
    v['cap'] = cv2.VideoCapture(v['fn'])
    v['open'] = v['cap'].isOpened()
    v['data'] = []

    retry = 5 # five failure retry is accepted before stop reading due to failure
    while True:
        ok, f = v['cap'].read() # read the capture
        if not ok:
            if retry > 0:
                retry -= 1
                continue
            break
        v['data'].append(f) # accumulate frame into v['data'] (it's a list)
    return v

def getdata(p, v):
    """
    the accumulation function

    usage: "framelist = reduce(getdata, v)"

    frames are re-arranged into {'frame': f, 'tag': t} dict, accumulated into "p" (named after "previous value")
    """
    p.extend(map(lambda vi: {'frame': vi, 'tag': v['tag'] if 'tag' in v else v['fn']}, v['data']))
    return p

def mkdir(p):
    """
    mkdir if it not exists
    """
    if not os.path.isdir(p):
        os.mkdir(p)

def extract(videos, fn, name, shuffle=True):
    """
    process video structures from "videos", and save as "fn" 

    note that the dataset will be sliced to reduce the size in each pickled data
    """
    extracted = list(map(processor, videos))

    for p in extracted:
        print('{}: {}: {} frame(s)'.format(name, p['fn'], len(p['data'])))

    # reduce(func, iterable[, initializer])
    # The reduce() function accumulates the elements in the parameter sequence.
    #
    # The reduce function performs the following operations on all the data in a data set (linked list, tuple, etc.):
    #  use the function func passed to reduce() to operate on the first two elements in the set, and then use the result to operated with the third data using the func function, and finally a result is obtained.
    #
    # ```
    # def reduce(func, iterable, initializer):
    #     iv = initializer
    #     for i in iterable:
    #         iv = func(iv, i)
    #     return iv
    # ```
    data = reduce(getdata, extracted, [])

    # shuffle the dataset
    if shuffle:
        random.shuffle(data)

    index = [] # dataset index
    i = 0
    for k in range(0, len(data), 128):
        part = data[k : k+128]
        i += 1

        # construct file name
        part_fn = fn + '.part{}.pickle'.format(i)

        # construct index item
        index.append({'fn': part_fn, 'len': len(part)})

        # save partial data into part file
        with open(part_fn, 'wb') as f:
            pickle.dump(part, f)

    # dump index file
    with open(fn + '.index.pickle', 'wb') as f:
        pickle.dump(index, f)

# generate trainset
extract(train_videos, 'trainset', 'train')

# generate testset
extract(test_videos, 'testset', 'test', False)
