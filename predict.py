import argparse
import serial
import socket
import time
import datetime
import json
import simplejson
import collections
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
from multiprocessing import Process

import cv2
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import Variable

cameraNum = 4

nonConce_rates = []
average_nonConces = []
jsonData = []


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
    return highData, lowData, highData_idx, lowData_idx


def latest_filePath(directory):
    target = os.path.join(directory, '*')
    files = [(f, os.path.getmtime(f)) for f in glob(target)]
    latest_filePath = sorted(files, key = lambda files: files[1])[-1]
    return latest_filePath[0]


def feedbackFunc(folder_path, camera, channel, model, nonConce, average, json):
    latest_folder = latest_filePath(folder_path + str(camera) + '/')
    _, images = dataRead(latest_folder + '/')
    counter = 0
    highResult_total = 0

    for image in images:
        t = int(time.mktime(datetime.datetime.now().timetuple()))
        #predict
        highResult, lowResult, highResult_idx, lowResult_idx = predict_result(image, channel, model)
        if highResult_idx == 1:
            counter += 1
            highResult_total += highResult

    nonConce_rate = counter/len(images)
    average_nonConce = (lambda a,b: b == 0 and 1 or a/b)(highResult_total, counter)
    each_jsonData = {"time":t, "rate":nonConce_rate, "value":average_nonConce}

    nonConce.append(nonConce_rate)
    average.append(average_nonConce)
    json.update(each_jsonData)

    return


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/publish')
def publish():

    # configuration of connecting to arduino
    udp_ip = "192.168.11.9"
    udp_port = 8888
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


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

        while True:
            jsonData = []
            for camera in range(cameraNum):
                each_jsonData = collections.OrderedDict()
                latest_folder = latest_filePath(args.output + str(camera) + '/')
                _, images = dataRead(latest_folder + '/')
                counter = 0
                highResult_total = 0
                switch = False

                for image in images:
                    t = int(time.mktime(datetime.datetime.now().timetuple()))
                    #predict
                    highResult, lowResult, highResult_idx, lowResult_idx = predict_result(image, args.channel, model)
                    if highResult_idx == 1:
                        counter += 1
                        highResult_total += highResult

                nonConce_rate = float(counter)/len(images)
                average_nonConce = (lambda a,b: b == 0 and 1 or a/b)(highResult_total, counter)
                #each_jsonData = {"time":t, "y":average_nonConce}
                each_jsonData["time"] = t
                each_jsonData["y"] = nonConce_rate
                jsonData.append(each_jsonData)

                if nonConce_rate >= 0.5:
                    threshold = "HIGH"
                    switch = True
                    if average_nonConce >= 0.7 and switch == True:
                        threshold = "average_HIGH"
                else:
                    threshold = "LOW"
                    switch = False

                sock.sendto(threshold, (udp_ip,udp_port))

            ws.send(simplejson.dumps(jsonData))
            time.sleep(3)

            # multi processing
            '''
            global nonConce_rates, average_nonConces, jsonData
            switch = False

            camera0feed = Process(target = feedbackFunc,
                                  args = (args.output, 0, args.channel, args.model, nonConce_rates, average_nonConces, jsonData))
            camera1feed = Process(target = feedbackFunc,
                                  args = (args.output, 1, args.channel, args.model, nonConce_rates, average_nonConces, jsonData))

            camera0feed.start()
            camera0feed.join()

            camera1feed.start()
            camera1feed.join()

            for idx, value in enumerate(nonConce_rates):
                if value >= 0.5:
                    threshold = "HIGH"
                    switch = True
                else:
                    threshold = "LOOW"
                sock.sendto(threshold, (udp_ip,udp_port))
                ws.send(json.dumps(jsonData))

                if switch == True and average_nonConces[idx] >= 0.7:
                    threshold = "average_HIGH"
                    sock.sendto(threshold, (udp_ip,udp_port))

            '''

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='face prediction')
    parser.add_argument('--testPath', '-t', type=str, default='./datasets/test/')
    parser.add_argument('--detect', '-d', type=str, default='')
    parser.add_argument('--output', '-o', type=str, default='Y:/experiment_1226/predict/') # win to ubuntu
    #parser.add_argument('--output', '-o', type=str, default='/volumes/share/pre_experiment/predict/') # mac to ubuntu
    #parser.add_argument('--output', '-o', type=str, default='./datasets/pre_experiment/predict/') # local
    parser.add_argument('--size', '-s', type=int, default=128)
    parser.add_argument('--model', '-m', type=str, default='output3_20161217232915.model')
    parser.add_argument('--channel', '-c', type=int, default=3)

    args = parser.parse_args()

    #load model and translate it from gpu model to cpu model
    model = L.Classifier(HoseiCNN())
    chainer.serializers.load_npz(args.model, model)
    model.to_cpu()

    app.debug = True
    server = pywsgi.WSGIServer(('localhost',8000), app, handler_class=WebSocketHandler)
    server.serve_forever()
