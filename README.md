## README


We split the classification of hand gesture in egocentric videos into three separate steps, using three end-to-end models to complete the function, including an SSD[ref]-based model to detecting hands, a detection algorithm[ref] extracted from openpose[ref] to locating hand keypoints, and a simple fully-connected network to classify hand gesture. We train the model on the EgoHands[ref] dataset and a home-made dataset. (what does the home-made dataset consist of) Since the method we use is based on frames and does not rely on temporal information in the first two steps, we record videos of each gesture separately.


##### Hand Detection

In this step, we locate the hand (retrive the bounding box) and extract the image clip that focus on the hand. An object detection model[ref]trained (transfered) on the EgoHands dataset.

###### code [has problem]:

Faster-RCNN

1）gethanddetector.py: a failed hand detector (#1)

SSD

1) TO BE TEST: EXTERNAL: https://github.com/zllrunning/hand-detection.PyTorch: The hand detector for which I failed to setup up env. It contains a pretrained model (described in its readme), and the `nms` in its folder fails to compile on windows.

2) handtracking folder: a failed hand detector (#2), a network that use tensorflow. This folder is a git repository. (https://github.com/victordibia/handtracking.git)


##### Hand Keypoint Detection

Hand keypoints are extracted to reduce as much interference as possible. We choose the hand keypoint detector from OpenPose. MMdnn is used to convert the original model, which is written in caffe, into a PyTorch model. We neither modified or (re-) trained the model.

###### code:

gethandpos.py: calculate hand keypoints frame by frame for one part of the dataset (thus, you can process the whole dataset in parallel)

##### Hand Gesture Classification

After being normalized (make the center (mean) of all coordinates at (0,0)), the matrix consisting of input coordinates of hand keypoints is flattened and passed into a fully-connected network with three hidden layers with 128, 256, 72 hidden units, respectively, using ReLU activation. The final regression is linear.

###### code:

analysis.py: train a fully-connected network to classify gesture frame by frame

test.py: generate test videos (using the trained network)

##### others

useful：


demo2.py: Use hand_pose_model to show hand keypoints

mkdataset.py: extract dataset from tagged videos (see file content)

reference：

src/ && model/: the hand_pose_model (downloaded from github)

ego_model/: the trained model in caffe and the translated pytorch model, used by the failed hand detector#1. The conversion is conducted by MMdnn, the code is hand-modified after automatic conversion.

models-1.12.0/: tensorflow/models, the repo that contains Objection Detection API, which is used by the failed hand detector #2

