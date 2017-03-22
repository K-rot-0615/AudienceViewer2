import argparse
import chainer
from chainer.links.caffe import CaffeFunction
import _pickle as pickle

loadPath = './lib/bvlc_alexnet.caffemodel'
savePath = './models/alexnet.pkl'
caffeModel = CaffeFunction(loadPath)
pickle.dump(caffeModel, open(savePath, 'wb'))