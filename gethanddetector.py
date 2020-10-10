#!/usr/bin/env python

from typing import *
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

import random

import torchvision
from torchvision.models.detection.faster_rcnn import FasterRCNN
from torchvision.models.detection.mask_rcnn import MaskRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from scipy.io import loadmat

from ego_model.hand_type_classifier import HandTypeClassifier

from src import model
from src import util
from src.hand import Hand

import os
import sys

import cv2
import torch
from torch import nn
from torch import optim
import time

# system environmental variable 'SEPARATE_TRAINING' is also a input, indicate to backward multiple type of losses instead of one type during training if this environmental variable is present (not None)


# whether to load everything into memory before training
# enable this option can cause the usage of memory (or GPU memory) out of its physical constraint (and cause out-of-memory) but with accelerate training a lot
MEMORY_CACHE=True


def get_bounding_rect(polygon):
    """
    calculate the bounding rect of polygon

    polygon: 2d array, list of x-y coordinate tuple (but can be numpy form)

    returns (x_min, y_min, x_max, y_max)
    """
    x1, y1, x2, y2 = float('inf'), float('inf'), float('-inf'), float('-inf')
    for x, y in polygon:
        if x < x1:
            x1 = x
        if y < y1:
            y1 = y
        if x > x2:
            x2 = x
        if y > y2:
            y2 = y
    return x1, y1, x2, y2

# the meaning (label) of 0,1,2,3 in EgoHands dataset
orig_labels = ['myleft', 'myright', 'yourleft', 'yourright']

def load_egohands_dataset(root: str):
    """
    extract every frame with its positional information from EgoHands dataset in the specified folder

    PARAMETERS:
        root: str
            root folder of EgoHands directory

    this is a generator function, you may run `list(load_egohands_dataset(root))` to get the list

    yield {'file': f, 'scene': s, 'boxes', b, 'labels': l}
    return None
    """

    # iterate over all sub-directory in root
    for dir_name in os.listdir(root):
        path = os.path.join(root, dir_name)
        if os.path.isdir(path):
            # path is the sub-directory of root
            # check the presence of polygons.mat in the directory
            full_path = os.path.join(path, 'polygons.mat')
            if os.path.isfile(full_path):
                # get the list of frames, which is all file in the directory with "frame_" and ".jpg" in the file name
                # we don't have to make this a list, since sorting on the iterable is acceptable
                frames = filter(lambda fn: 'frame_' in fn and '.jpg' in fn, os.listdir(path))
                # os.listdir list file with correct order only on some platforms, so we have to sort it to make sure the rank is correct
                frames = sorted(frames)

                # we treat sub-directory name in root as the scene name
                scene = dir_name

                # load all polygons, and change its format into what we want (3-d array)
                polygons = loadmat(full_path)['polygons'][0]
                polygons = np.stack([ polygons[label] for label in orig_labels ], axis=1)

                # co-iterate frame and polygon
                # if len(frames) and len(polygons) are not the same, exception will be thrown
                for framedata in zip(frames, polygons):

                    # retrive frame-polygon pair
                    f, p = framedata
                    f = os.path.join(path, f) # build full path of frame

                    # calculate bounding rect of each polygon (we do not use MaskRCNN so the rectangle region should work)
                    boxes = []
                    labels = []
                    for label_id in range(len(orig_labels)):
                        label_name = orig_labels[label_id]
                        if p[label_id].shape[1] != 0:
                            boxes.append(torch.tensor(get_bounding_rect(p[label_id].squeeze()), dtype=torch.float))
                            labels.append(label_id)
                    
                    # if we store image in memory, load image now
                    if MEMORY_CACHE:
                        f = cv2.imread(f)
                        f = torch.from_numpy(f).permute((2, 0, 1)).float() # change shape into (band, width, height)
                    
                    # if we have a box in this frame, show it
                    if len(boxes) > 0:
                        yield { 'file': f, 'scene': scene, 'boxes': torch.stack(boxes), 'labels': torch.tensor(labels, dtype=torch.int64) }
            else:
                print('Warning: {} does not exist.'.format(full_path))
    return


########## Load dataset
print('loading dataset...')
dataset = list(load_egohands_dataset('egohands/_LABELLED_SAMPLES'))
random.shuffle(dataset)


########## TEST CODE: Show each frame to verify the loading code of EgoHands dataset is correct
# k = dataset[0]
# print(k)
# img = cv2.imread(k['file'])
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# for box, label in zip(k['boxes'], k['labels']):
#     img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
#     img = cv2.putText(img, orig_labels[label], (box[0] + 5, box[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2, cv2.LINE_AA)
# plt.imshow(img)
# plt.axis('off')
# plt.show()
# exit(1)

test_image = 'demo2.jpg'


########## TEST CODE: When launch in ipython, choose demo2.jpg automatically
# if 'ipykernel_launcher' in sys.argv[0]:
#     test_image = 'demo2.jpg'
# else:
#     test_image = None
#     if len(sys.argv) != 2:
#         print('Usage: {} <file>'.format(sys.argv[0]))
#         exit(1)
#     else:
#         test_image = sys.argv[1]

def test():
    """
    Show prediction result using the first frame in the dataset
    """

    # load image and adjust its format
    if MEMORY_CACHE:
        test_input = dataset[0]['file']
        oriImg = test_input.byte().permute((1, 2, 0)).numpy()  # B,G,R order
    else:
        oriImg = cv2.imread(dataset[0]['file'])  # B,G,R order
        test_input = torch.from_numpy(oriImg).permute((2, 0, 1)).float()
    
    # transfer data on GPU on demand
    if CUDA:
        test_input = test_input.cuda()

    # perform prediction
    net.eval()
    with torch.no_grad():
        result = net(test_input.unsqueeze(0))[0]

    print(result)

    # draw rectangles and its class
    img = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    for box, label, score in zip(result['boxes'], result['labels'], result['scores']):
        # if score > 0.5:
        if label < len(orig_labels):
            img = cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 3)
            img = cv2.putText(img, '{}: {:.0%}'.format(orig_labels[label], score), (box[0] + 5, box[3] - 5), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 0), 2, cv2.LINE_AA)
    plt.imshow(img)
    plt.axis('off')
    plt.show()


#### load pretrained model and initialize FasterRCNN
print('loading model')
backbone = HandTypeClassifier('ego_model/hand_type_classifier.npy').features
backbone.out_channels = 256
# backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# backbone.out_channels = 1280

#### The anchor generator
anchor_generator = AnchorGenerator(
    sizes=((32, 64, 128, 256, 512),),
    aspect_ratios=((0.5, 1.0, 2.0),)
    )

#### Use FasterRCNN
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=7, sampling_ratio=2)
net = FasterRCNN(backbone, num_classes=4, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
# mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'], output_size=14, sampling_ratio=2)
# net = MaskRCNN(backbone, num_classes=4, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler, mask_roi_pool=mask_roi_pooler)

########## Training parameter

# set model name, epoch count and SEPARATE_TRAINING parameter according to the presence of the environment variable
if os.environ.get('SEPARATE_TRAINING') is not None:
    SEPARATE_TRAINING=False
    MODEL_NAME='hand-detector-joinedloss.pth'
    EPOCH = 6
else:
    SEPARATE_TRAINING=True
    MODEL_NAME='hand-detector.pth'
    EPOCH = 24

BATCH = 8 # train batch size

SINGLE_LOSS=None # only backward this loss, if SINGLE_LOSS is not None. Or, automatically choose the loss
# possible values: 'loss_box_reg', 'loss_rpn_box_reg', 'loss_objectness', 'loss_classifier', None

CUDA = True and torch.cuda.is_available() # Enable CUDA when possible

CONTINUE_TRAIN=True # Continue to train a existed model (instead of showing testing results)



########## the code for training

# printing out parameters
print('separate training: {}'.format(SEPARATE_TRAINING))
print('output model: {}'.format(MODEL_NAME))

# transfer model to GPU if enabled
if CUDA:
    net = net.cuda()
xnet = net.module if 'module' in dir(net) else net

# load pretrained model
pretrained = None
epoch_offset = 0
if os.path.isfile(MODEL_NAME):
    pretrained = torch.load(MODEL_NAME)
    xnet.load_state_dict(pretrained['net'])
    epoch_offset = pretrained['epoch']

if pretrained is None or CONTINUE_TRAIN:
    # train the model, initialize optimizer
    print('training started from epoch {}'.format(epoch_offset))
    net.train()
    opt = optim.Adagrad(net.parameters(), lr = 0.01)
    if pretrained:
        opt.load_state_dict(pretrained['opt'])

    # the training
    for epoch in range(EPOCH):
        # the container for batch training
        batched_x = []
        batched_target = []

        i = 0 # current item in the training set, used to estimate training time
        
        st = time.time()
        nt = st
        for item in dataset:
            i += 1
            
            # load the image (either from memory or from file)
            if MEMORY_CACHE:
                x = item['file']
            else:
                img = cv2.imread(item['file'])
                x = torch.from_numpy(img).permute((2, 0, 1)).float()

            # transfer to GPU
            if CUDA:
                x = x.cuda()

            # append to batched_### for batch processing
            batched_x.append(x)
            if CUDA:
                batched_target.append({
                    'boxes': item['boxes'].cuda(),
                    'labels': item['labels'].cuda()
                })
            else:
                batched_target.append({
                    'boxes': item['boxes'],
                    'labels': item['labels']
                })

            # if one batch is assembled, train it!
            if len(batched_target) == BATCH:
                # bundle the input into one tensor
                x = torch.stack(batched_x, 0)
                losses = net(x, batched_target)

                # backward the loss
                if SINGLE_LOSS is not None:
                    opt.zero_grad()
                    losses[SINGLE_LOSS].backward()

                elif SEPARATE_TRAINING:
                    opt.zero_grad()

                    # choose the loss automatically
                    if epoch <= float(EPOCH) / 2:
                        losses['loss_rpn_box_reg'].backward()
                    else:
                        losses['loss_classifier'].backward()

                    # if epoch <= float(EPOCH) / 3:
                    #     losses['loss_box_reg'].backward()
                    # elif epoch <= float(EPOCH) / 3 * 2:
                    #     losses['loss_rpn_box_reg'].backward()
                    # else:
                    #     losses['loss_classifier'].backward()

                    # if epoch <= float(EPOCH) / 4:
                    #     losses['loss_box_reg'].backward()
                    # elif epoch <= float(EPOCH) / 4 * 2:
                    #     losses['loss_rpn_box_reg'].backward()
                    # elif epoch <= float(EPOCH) / 4 * 3:
                    #     losses['loss_objectness'].backward()
                    # else:
                    #     losses['loss_classifier'].backward()
                else:
                    # use the sum of losses
                    loss_val = sum(loss for loss in losses.values())
                    opt.zero_grad()
                    loss_val.backward()
                
                # step the optimizer
                opt.step()

                # show the progress
                t = time.time()
                print('Epoch {}/{}: {}/{}, time: {:.2F}s, eta: {:.2F}s.'.format(epoch_offset + epoch + 1, epoch_offset + EPOCH, i, len(dataset), t - st, (len(dataset) - i) * (0.7 * (t - nt) / len(batched_x) + 0.3 * (t - st) / i)), end='   \r', flush=True)
                batched_target.clear()
                batched_x.clear()
                nt = t

        # complete one epoch
        
        #### TEST CODE: show test result after each epoch
        # test()
        # net.train()
        # nt = time.time()

        print('Epoch {}/{}: {}/{}, time: {:.2F}s.'.format(epoch_offset + epoch + 1, epoch_offset + EPOCH, i, len(dataset), nt - st), end='   \r', flush=True)
    print('done')

    # save the trained model
    torch.save({
        'opt': opt.state_dict(),
        'net': xnet.state_dict(),
        'epoch': epoch_offset + EPOCH,
        }, MODEL_NAME)


# test the model
test()

