import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


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
