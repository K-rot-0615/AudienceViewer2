# -*- coding:utf-8 -*-

import argparse
import numpy as np
import glob
from PIL import Image
from chainer.datasets import tuple_dataset


def labeling(data, channel):
    testData = []
    for n in data:
        path = n[0]
        label = n[1]
        imgList = glob.glob(path + '*')
        for imgName in imgList:
            testData.append([imgName, label])
    testData = np.random.permutation(testData)

    imageData = []
    labelData = []

    if channel == 1:
        for n in testData:
            path = n[0]
            label = n[1]
            imgList = glob.glob(path + '*')
            for imgName in imgList:
                img = Image.open(imgName)
                imgData = np.asarray([np.float32(img)/255.0])
                lblData = label

                imageData.append(imgData)
                labelData.append(np.int32(lblData))

        threshold = np.int32(len(imageData)/10*8)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test = tuple_dataset.TupleDataset(imageData[threshold:], labelData[threshold:])

    else:
        # path and label
        for n in testData:
            path = n[0]
            label = n[1]
            imgList = glob.glob(path + '*')
            for imgName in imgList:
                img = Image.open(imgName)
                r,g,b = img.split()
                imgData_R = np.asarray(np.float32(r)/255.0)
                imgData_G = np.asarray(np.float32(g)/255.0)
                imgData_B = np.asarray(np.float32(b)/255.0)

                imgData = np.asarray([imgData_R,imgData_G,imgData_B])
                lblData = label

                imageData.append(imgData)
                labelData.append(np.int32(lblData))

        threshold = np.int32(len(imageData)/10*8)
        train = tuple_dataset.TupleDataset(imageData[0:threshold], labelData[0:threshold])
        test = tuple_dataset.TupleDataset(imageData[threshold:], labelData[threshold:])

    return train, test


def getPredictData(image, channel):
    if channel == 1:
        img = Image.open(image)
        imgData = np.asarray([np.float32(img)/255.0])
        return imgData

    else:
        img = Image.open(image)
        r,g,b = img.split()
        imgData_R = np.asarray(np.float32(r)/255.0)
        imgData_G = np.asarray(np.float32(g)/255.0)
        imgData_B = np.asarray(np.float32(b)/255.0)
        imgData = np.asarray([[[imgData_R, imgData_G, imgData_B]]])

        return imgData


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--channel', '-c', type=int, default=3)
    args = parser.parse_args()

    data = []
    data.append(np.asarray(['./datasets/ryota/', 0]))
    data.append(np.asarray(['./datasets/masakatsu/', 1]))
    data.append(np.asarray(['./datasets/sakamoto/', 2]))
    train, test = labeling(data, args.channel)
    #print len(train)
    #print test[10]
