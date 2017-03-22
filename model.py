import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class Alex(chainer.Chain):
    """Single-GPU AlexNet without partition toward the channel axis."""

    insize = 128

    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(None, 96, 8, stride=4),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 384, 3, pad=1),
            conv4=L.Convolution2D(None, 256, 3, pad=1),
            conv5=L.Convolution2D(None, 32, 3, pad=1),
            my_fc6=L.Linear(None,144),
            my_fc7=L.Linear(None,50),
            my_fc8=L.Linear(None,3),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(F.relu(self.conv5(h)), 3, stride=2)
        h = F.dropout(F.relu(self.my_fc6(h)), train=self.train)
        h = F.dropout(F.relu(self.my_fc7(h)), train=self.train)
        h = self.my_fc8(h)

        return h


class CNN(chainer.Chain):

    insize = 128

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(None, 128, 6, stride=2),
            conv2=L.Convolution2D(None, 256, 5, stride=2, pad=1),
            conv3=L.Convolution2D(None, 384, 3, stride=2, pad=1),
            conv4=L.Convolution2D(None, 256, 3, pad=1),
            conv5=L.Convolution2D(None, 256, 3, pad=1),
            fc6=L.Linear(None,50),
            fc7=L.Linear(None,3),
        )
        self.train = True

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))
        h = F.relu(self.fc6(h))
        h = self.fc7(h)

        return h


## hosei uiniversity ver.
class HoseiCNN(chainer.Chain):

    insize = 128

    def __init__(self):
        super(HoseiCNN, self).__init__(
            conv1=L.Convolution2D(None, 128, 6, pad=2, stride=2),
            conv2=L.Convolution2D(None, 256, 5, pad=2),
            conv3=L.Convolution2D(None, 256, 4, pad=1),
            fc4=L.Linear(None,3),
        )
        self.train = True

    def __call__(self, x):
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv1(x))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv2(h))), 2, stride=2)
        h = F.max_pooling_2d(F.local_response_normalization(
            F.relu(self.conv3(h))), 1, stride=2)
        h = self.fc4(h)

        return h