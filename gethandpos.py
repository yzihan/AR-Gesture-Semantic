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


# Process xxxxxxx.part#.pickle (part of the dataset) instead of the whole dataset
# this design can accelerate the overall pose estimation step by parallel processing
# you can start multiple instance of this program on different part in the dataset
#
# e.g.
#   (In terminal 1): python gethandpos.py test.part1.pickle
#   (In terminal 2): python gethandpos.py test.part2.pickle
#   (In terminal 3): python gethandpos.py test.part3.pickle
#   ...
# 
# You may also use HPC job scheduler, e.g. slurm or LSF, to dispatch the task :)
# the "parallel" utility also helps


# the model
hand_estimation = Hand('model/hand_pose_model.pth')

def label_handpos(fn):
    """
    run hand pose estimation on each frame in the video file, save back to the original file
    """

    # load pickle file (xxxxxxx.part#.pickle)
    print('Opening {}...'.format(fn))
    with open(fn, 'rb') as f:
        data = pickle.load(f)

    # print out data metrics
    print('{}: {} frame(s)'.format(fn, len(data)))
    print('Labeling...')

    st = time.time()
    nt = st

    # iterate over all the frames
    tot = len(data)
    cur = 0
    for frame_data in data:
        # do estimation
        frame_data['hand'] = hand_estimation(frame_data['frame'])
        t = time.time()

        # calculate estimated finish time
        cur += 1
        eta = (tot - cur) * (0.3 * (t - st) / cur + 0.7 * (t - nt))
        print('Processing... {}/{}   time elapsed: {:.1f}s   time remaining: {:.1f}s'.format(cur, tot, t - st, eta), end='  \r', flush=True)
        nt = t

    # save back to the original file
    print('\nSaving back to {}...'.format(fn))
    with open(fn, 'wb') as f:
        pickle.dump(data, f)


# process parameter
test_image = None
if len(sys.argv) != 2:
    # should process (xxxxxxx.part#.pickle)
    print('Usage: {} <file>'.format(sys.argv[0]))
    exit(1)
else:
    test_image = sys.argv[1]

# do labeling and save back
label_handpos(test_image)
