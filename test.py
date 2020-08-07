#!/usr/bin/env python

import numpy as np
import pickle
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

def load_dataset(fn):
    with open(fn + '.index.pickle', 'rb') as f:
        index = pickle.load(f)
    data = []
    for item in index:
        with open(item['fn'], 'rb') as f:
            part_data = pickle.load(f)
        if len(part_data) != item['len']:
            print('Warning: incorrect index file: len({}) != {}'.format(item['fn'], item['len']))
        data.extend(part_data)
    return data

test = load_dataset('testset')

print('test:', len(test))

def normalize_hand(hand):
    retv = hand.astype(np.float)
    nonzero = (retv == 0).sum(1) != 2
    retv[nonzero] -= np.average(retv[nonzero], 0)
    return retv




import torchvision.models as models

class mynet(nn.Module):
    def __init__(self, shp, l):
        super(mynet, self).__init__()
        self.thenet = nn.Sequential(
            nn.Linear(np.product(shp), 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, 72), nn.ReLU(True),
            nn.Linear(72, l),
            )
    def forward(self, X):
        v = X.flatten(start_dim=1)
        v = self.thenet(v)
        return v

obj = torch.load('train-result.pth')
Y_map = obj['Y_map']
EPOCH = obj['epoch']

# net = models.densenet161()
net = mynet(test[0]['hand'].shape, len(Y_map))
loss = nn.CrossEntropyLoss()
opt = optim.Adagrad(net.parameters(), lr = 0.01)

opt.load_state_dict(obj['opt'])
net.load_state_dict(obj['net'])

print('Model at epoch #{}'.format(EPOCH))


def __linear_gradient(pos, c1, c2):
    return ((c2 - c1) * pos + c1 if pos > 0 else c1) if pos < 1 else c2

def linear_gradient(pos, c1, c2):
    return (__linear_gradient(pos, c1[0], c2[0]), __linear_gradient(pos, c1[1], c2[1]), __linear_gradient(pos, c1[2], c2[2]))

import cv2
from src import util

video_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = 30
output_video = None
output_file = None
output_file = 'result.mp4'

max_seq = len(test)
seq = 0
for i in test:
    seq += 1
    img = i['frame']
    hand = i['hand']
    
    x = normalize_hand(hand)

    canvas = img.copy()
    canvas = util.draw_handpose(canvas, [hand])

    rect = (0, 0, canvas.shape[1], canvas.shape[0])
    if output_video is None and output_file is not None:
        output_video = cv2.VideoWriter(output_file, video_fourcc, fps, (canvas.shape[1], canvas.shape[0]))

    with torch.no_grad():
        predicted = net(torch.from_numpy(x).float().unsqueeze_(0)).squeeze()
        predicted = F.softmax(predicted, 0)
        predicted_class = predicted.argmax()
        predicted_prob = predicted[predicted_class]
        color = linear_gradient((predicted_prob.item() - 0.5) / 0.4, (31, 15, 197), (14, 161, 19))
        output = '{}, prob={:.0%}'.format(Y_map[predicted_class], predicted_prob)
        if predicted_prob <= 0.5:
            output = 'Undetermined'
        cv2.putText(canvas, output, (rect[0] + 10, rect[3] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1., color, 2, cv2.LINE_AA)

    print('Frame {}/{}'.format(seq, max_seq), end='  \r', flush=True)

    if output_video is not None:
        output_video.write(canvas)
    else:
        cv2.imshow('img', canvas)
        if cv2.waitKey(1) == 27:
            break
cv2.destroyAllWindows()
if output_video is not None:
    output_video.release()
print('')
