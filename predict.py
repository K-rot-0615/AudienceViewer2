import argparse
import serial
import socket
import time
import datetime
import json
import os
import random
import numpy as np
from gatherData import dataRead, latest_filePath
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
    y = F.softmax(model.predictor(x.datum[0]))
    return y.datum[0]


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


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/publish')
def publish():
    '''
    # configuration of connecting to arduino
    udp_ip = "192.168.11.14"
    udp_port = 8888
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    '''

    if request.environ.get('wsgi.websocket'):
        ws = request.environ['wsgi.websocket']

        '''
        # analyze using already detected data
        image = dataRead(args.output)
        for image in images:
            #t = int(time.mktime(datetime.datetime.now().timetuple()))
            # predict
            highResult, lowResult, highResult_idx, lowResult_idx = predict_result(image, args.channel, model)
            #when the concentration is declined
            #if highResult_idx == 1:
            ws.send(json.dumps([{"time":t, "y":lowResult * 100},{"time":t, "y":highResult * 100}]))
            time.sleep(1)
        '''

        '''
        # analyze using realtime data
        while True:
            for camera in cameraNum:
                image = latest_filePath(args.output + '/' + camera)
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
                #sock.sendto(threshold, (udp_ip,udp_port))
                time.sleep(1)
        '''

        while True:
            latest_folder = latest_filePath(args.output + str(0) + '/')
            _, images = dataRead(latest_folder + '/')
            for image in images:
                t = int(time.mktime(datetime.datetime.now().timetuple()))
                #predict
                highResult, lowResult, highResult_idx, lowResult_idx = predict_result(image, args.channel, model)
                #print highResult, highResult_idx
                ws.send(json.dumps([{"time":t, "y":lowValue * 5},
                                {"time":t, "y":highValue * 5}]))

    return


if __name__ == '__main__':

    app.debug = True
    server = pywsgi.WSGIServer(('localhost',8000), app, handler_class=WebSocketHandler)
    server.serve_forever()
