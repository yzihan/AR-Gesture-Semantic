#!/usr/bin/env python
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

#from src import util
from src.hand import Hand

import sys
import pickle

import time

hand_estimation = Hand('model/hand_pose_model.pth')

def label_handpos(fn):
    print('Opening {}...'.format(fn))
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    print('{}: {} frame(s)'.format(fn, len(data)))
    print('Labeling...')

    st = time.time()
    nt = st

    tot = len(data)
    cur = 0
    for frame_data in data:
        frame_data['hand'] = hand_estimation(frame_data['frame'])
        t = time.time()
        cur += 1
        eta = (tot - cur) * (0.3 * (t - st) / cur + 0.7 * (t - nt))
        print('Processing... {}/{}   time elapsed: {:.1f}s   time remaining: {:.1f}s'.format(cur, tot, t - st, eta), end='  \r', flush=True)
        nt = t

    print('\nSaving back to {}...'.format(fn))
    with open(fn, 'wb') as f:
        pickle.dump(data, f)


test_image = None
if len(sys.argv) != 2:
    print('Usage: {} <file>'.format(sys.argv[0]))
    exit(1)
else:
    test_image = sys.argv[1]

label_handpos(test_image)
