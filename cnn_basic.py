import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):

    insize = 128

    def __init__(self):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(None, 96, 8, stride=2),
            conv2=L.Convolution2D(None, 256, 5, stride=2, pad=1),
            conv3=L.Convolution2D(None, 384, 3, stride=2, pad=1),
            conv4=L.Convolution2D(None, 256, 3, pad=1),
            fc5=L.Linear(None,100),
            fc6=L.Linear(None,30),
            fc7=L.Linear(None,3),
        )
        self.train = True

    def __call__(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.dropout(F.relu(self.fc5(h)), train=self.train)
        h = F.dropout(F.relu(self.fc6(h)), train=self.train)
        h = self.fc7(h)

        return h
