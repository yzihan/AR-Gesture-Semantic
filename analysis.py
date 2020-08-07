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

train = load_dataset('trainset')

print('train:', len(train))

cv_split_point = int(len(train) * 0.8)
train, cv_test = train[:cv_split_point], train[cv_split_point:]

print('cv_train:', len(train))
print('cv_test:', len(cv_test))

def normalize_hand(hand):
    retv = hand.astype(np.float)
    nonzero = (retv == 0).sum(1) != 2
    retv[nonzero] -= np.average(retv[nonzero], 0)
    return retv

def batch_dataset(d, BATCH_SIZE, has_Y, Y_map = None):
    X = []
    if has_Y:
        Y = []
        new_map = set(map(lambda v: v['tag'], d))
        if Y_map is not None:
            Y_map.extend(new_map - set(Y_map))
        else:
            Y_map = list(new_map)

    for i in range(0, len(d), BATCH_SIZE):
        part_data = d[i: i + BATCH_SIZE]
        X.append(torch.tensor(list(map(lambda v: normalize_hand(v['hand']), part_data)), dtype=torch.float32))
        if has_Y:
            Y.append(torch.tensor(list(map(lambda v: Y_map.index(v['tag']), part_data))))

    if has_Y:
        return X, Y, Y_map
    else:
        return X


import torchvision.models as models

class mynet(nn.Module):
    def __init__(self, shp, class_num):
        super(mynet, self).__init__()
        self.thenet = nn.Sequential(
            nn.Linear(np.product(shp), 128), nn.ReLU(True),
            nn.Linear(128, 256), nn.ReLU(True),
            nn.Linear(256, 72), nn.ReLU(True),
            nn.Linear(72, class_num),
            )
    def forward(self, X):
        v = X.flaten(start_dim=1)
        v = self.thenet(v)
        return v


EPOCH = 20
BATCH_SIZE = 256
X, Y, Y_map = batch_dataset(train, BATCH_SIZE, True)
cv_X, cv_Y, Y_map = batch_dataset(cv_test, BATCH_SIZE, True, Y_map)

# net = models.densenet161()
net = mynet(X[0].shape[1:], len(Y_map))
loss = nn.CrossEntropyLoss()
opt = optim.Adagrad(net.parameters(), lr = 0.01)

groundtruth_counter = [ np.sum([ y.eq(i).sum() for y in cv_Y ]) for i in range(len(Y_map)) ]

for i in range(1, EPOCH + 1):
    k = 0
    p = 0
    for x,y in zip(X, Y):
        predicted = net(x)
        loss_val = loss(predicted, y)
        opt.zero_grad()
        loss_val.backward()
        opt.step()
        k += loss_val.item() * x.shape[0]
    counter = [ 0 for i in range(len(Y_map)) ]
    correct = 0
    with torch.no_grad():
        for x,y in zip(cv_X, cv_Y):
            predicted = net(x)
            loss_val = loss(predicted, y)
            p += loss_val.item() * x.shape[0]
            predicted_class = predicted.argmax(1)

            predicted_success = predicted_class == y
            for y_val in range(len(Y_map)):
                counter[y_val] += np.logical_and(predicted_success, y == y_val).sum()
            correct += predicted_success.sum()
    k /= len(train)
    p /= len(train)
    acc = 1. * correct / len(cv_test)
    bal_acc = np.sum([ 1. * counter[x] / groundtruth_counter[x] for x in range(len(Y_map)) ]) / len(Y_map)
    print('epoch {}/{}: train loss={:.8F} test loss={:.8F} accuracy={:.5%} bal_acc={:.5%}'.format(i, EPOCH, k, p, acc, bal_acc))

torch.save({
    'opt': opt.state_dict(),
    'net': net.state_dict(),
    'Y_map': Y_map,
    'epoch': EPOCH,
    }, 'train-result.pth')

print('Model saved to {}'.format('train-result.pth'))
