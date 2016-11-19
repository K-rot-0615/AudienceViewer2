import argparse
import serial
import socket
import time
import datetime
import json
import os
import random
import numpy as np
from image import faceDetect
from image import faceDetect4Predict
from image import dataRead
from labeling import getPredictData
from alexnet import Alex

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


def predict_image(model, data):
    x = Variable(data)
    y = F.softmax(model.predictor(x.data[0]))
    return y.data[0]


def predict_result(image, channel, model):
    data = getPredictData(image, channel)
    result = predict_image(model, data)
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


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/publish')
def publish():
    parser = argparse.ArgumentParser(description='face prediction')
    parser.add_argument('--testPath', '-t', type=str, default='./datasets/test/')
    parser.add_argument('--detect', '-d', type=str, default='')
    parser.add_argument('--output', '-o', type=str, default='')
    parser.add_argument('--size', '-s', type=int, default=128)
    parser.add_argument('--model', '-m', type=str, default='output3_20161108213338.model')
    parser.add_argument('--channel', '-c', type=int, default=3)

    args = parser.parse_args()

    #load model and translate it from gpu model to cpu model
    model = L.Classifier(Alex())
    chainer.serializers.load_npz(args.model, model)
    model.to_cpu()

    # configuration of connecting to arduino
    udp_ip = "192.168.11.14"
    udp_port = 8888
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    if request.environ.get('wsgi.websocket'):
        ws = request.environ['wsgi.websocket']

        # analyze using already detected data
        '''
        images = dataRead(args.output)
        for image in images:
            t = int(time.mktime(datetime.datetime.now().timetuple()))
            # predict
            highResult, lowResult = predict_result(image, args.channel, model)

            ws.send(json.dumps([{"time":t, "y":lowResult * 100},
                                {"time":t, "y":highResult * 100}]))
            time.sleep(1)
        '''

        # analyze using realtime data
        while True:
            image = latest_filePath(args.output)
            t = int(time.mktime(datetime.datetime.now().timetuple()))
            # predict
            highValue, lowValue = predict_result(image, args.channel, model)
            midValue = highValue - lowValue
            ws.send(json.dumps([{"time":t, "y":lowValue * 5},
                                {"time":t, "y":highValue * 5}]))
            if midValue != 1:
                threshold = "LOOW"
            else:
                threshold = "HIGH"
            sock.sendto(threshold, (udp_ip,udp_port))
            time.sleep(1)
    return


if __name__ == '__main__':

    app.debug = True
    server = pywsgi.WSGIServer(('localhost',8000), app, handler_class=WebSocketHandler)
    server.serve_forever()
