import argparse
import serial
import socket
import time
import datetime
import json
import os
import random
import numpy as np
from gatherData import dataRead
from labeling import getPredictData
from alexnet import Alex
from cnn_hoseiuniv import HoseiCNN

#from smb.SMBConnection import SMBConnection
from glob import glob
from gevent import pywsgi
from geventwebsocket.handler import WebSocketHandler
from flask import Flask, request, render_template

import cv2
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

cameraNum = 4


def predict_image(model, datum):
    x = Variable(datum)
    y = F.softmax(model.predictor(x.data[0]))
    return y.data[0]


def predict_images(model, data):
    x = np.asarray(data)
    y = F.softmax(model.predictor(x))
    return y.data


def predict_result(image, channel, model):
    data = getPredictData(image, channel)
    result = predict_image(model, data)
    for idx, value in enumerate(result):
        if value == max(result):
            highData = value
            highData_idx = idx
        elif value == min(result):
            lowData = value
            lowData_idx = idx
    return (highData, lowData, highData_idx, lowData_idx)


def predict_results(images, channel, model):
    data = getPredictData(images, channel)
    result = predict_images(model, data)
    highData = max(result)
    lowData = min(result)
    return (highData, lowData)
    #print max(result)
    # print np.argmax(result)


def latest_filePath(directory):
    target = os.path.join(directory, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_filePath = sorted(files, key = lambda files: files[1])[-1]
    return latest_filePath[0]
    #print latest_filePath[0]


def latest_folderPath(directory):
    target = os.path.join(directory, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    print files


def main():
    parser = argparse.ArgumentParser(description='face prediction')
    parser.add_argument('--testPath', '-t', type=str, default='./datasets/test/')
    parser.add_argument('--detect', '-d', type=str, default='')
    #parser.add_argument('--output', '-o', type=str, default='Y:/predict_test/')
    parser.add_argument('--output', '-o', type=str, default='./datasets/pre_experiment/predict/')
    parser.add_argument('--size', '-s', type=int, default=128)
    parser.add_argument('--model', '-m', type=str, default='output3_20161217232915.model')
    parser.add_argument('--channel', '-c', type=int, default=3)

    args = parser.parse_args()

    #load model and translate it from gpu model to cpu model
    model = L.Classifier(HoseiCNN())
    chainer.serializers.load_npz(args.model, model)
    model.to_cpu()


    # realtime predict with multi camera
    '''
    while True:
        for camera in cameraNum:
            latest_folder = latest_filePath(args.output + '/' + camera + '/')
            _, images = dataRead(latest_folder + '/')
            print "next cameraNum is" + camera
            for image in images:
                t = int(time.mktime(datetime.datetime.now().timetuple()))
                # predict
                highResult, lowResult, highResult_idx, lowResult_idx = predict_result(image, args.channel, model)
                print camera, highResult, highResult_idx
    '''

    # realtime predict with single camera
    while True:
        latest_folder = latest_filePath(args.output + str(0) + '/')
        _, images = dataRead(latest_folder + '/')
        for image in images:
            t = int(time.mktime(datetime.datetime.now().timetuple()))
            #predict
            highResult, lowResult, highResult_idx, lowResult_idx = predict_result(image, args.channel, model)
            print highResult, highResult_idx

    return


if __name__ == '__main__':
    main()
