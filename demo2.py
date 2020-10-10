#!/usr/bin/env python
import cv2
import matplotlib.pyplot as plt
import copy
import numpy as np

from torchvision.models.detection.faster_rcnn import FasterRCNN
from scipy.io import loadmat

from src import model
from src import util
from src.body import Body
from src.hand import Hand

import sys

# load the model
hand_estimation = Hand('model/hand_pose_model.pth')

# read parameter
test_image = None
if len(sys.argv) != 2:
    print('Usage: {} <file>'.format(sys.argv[0]))
    exit(1)
else:
    test_image = sys.argv[1]

# read image
oriImg = cv2.imread(test_image)  # B,G,R order
canvas = copy.deepcopy(oriImg)

# do estimation
print('Estimating...')
peaks = hand_estimation(oriImg)

#peaks[:, 0] = np.where(peaks[:, 0]==0, peaks[:, 0], peaks[:, 0]+x)
#peaks[:, 1] = np.where(peaks[:, 1]==0, peaks[:, 1], peaks[:, 1]+y)

# draw handpose on the image
print('Rendering...')
canvas = util.draw_handpose(canvas, [peaks])

# generate output file
cv2.imwrite(test_image + '.png', canvas)

# preview the output file
plt.imshow(canvas[:, :, [2, 1, 0]])
plt.axis('off')
plt.show()
