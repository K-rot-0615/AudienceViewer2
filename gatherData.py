# -*- coding:utf-8 -*-

import argparse
import cv2
import time
import os
import os.path
import glob
from PIL import Image
from datetime import datetime
from multiprocessing import Process

camera = 3
cameraNum = [0, 1, 2]
cap = []


def dataRead(path):
	data = []
	imgList = glob.glob(path + '*')
	for imgName in imgList:
		data.append(imgName)
	return imgList, data


def gatherData(frame, cameraNum, pre_detectPath):
	now = datetime.now().strftime('%Y%m%d%H%M%S')
	now_image = pre_detectPath + now + '.png'
	if os.path.isdir(pre_detectPath) == False:
		os.mkdir(pre_detectPath)
		print 'make pre_detectPath!'
	cv2.imwrite(now_image, frame)


def faceDetect(pre_detectPath, savePath, resize):
	cascade_path = "./lib/haarcascade_frontalface_default.xml"
	cascade = cv2.CascadeClassifier(cascade_path)

	_, images = dataRead(pre_detectPath)
	for image in images:
		img = cv2.imread(image)
		facerect = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
		if len(facerect) > 0:
		    print 'Face has been detected'
		    if os.path.isdir(savePath) == False:
			    os.mkdir(savePath)
			    print 'make image path'
		    else:
			    print 'path already exists'

		i = 1
		imageList, _ = dataRead(savePath)
		for rect in facerect:
			x = rect[0] - 40
			y = rect[1] - 40
			width = rect[2] + 50
			height = rect[3] + 50
			dst = img[y : y + 10 + 3 * height, x : x + 30 + width]
			imgResize = cv2.resize(dst,(resize,resize))
			newImage_path = savePath + str(len(imageList) + i) + '.png'
			cv2.imwrite(newImage_path, imgResize)
			i = i + 1


def faceDetect4Predict(frame, cameraNum, pre_detectPath, savePath, resize):
	cascade_path = "./lib/haarcascade_frontalface_default.xml"
	cascade = cv2.CascadeClassifier(cascade_path)

	now = datetime.now().strftime('%Y%m%d%H%M%S')
	now_image = pre_detectPath + now + '.png'
	if os.path.isdir(pre_detectPath) == False:
		os.mkdir(pre_detectPath)
		print 'make pre_detectPath!'
	cv2.imwrite(now_image, frame)

	_, images = dataRead(pre_detectPath)
	for image in images:
		img = cv2.imread(image)
		facerect = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
		if len(facerect) > 0:
		    print 'Face has been detected'
		    if os.path.isdir(savePath) == False:
			    os.mkdir(savePath)
			    print 'make image path'
		    else:
			    print 'path already exists'

        i = 1
        imageList, _ = dataRead(savePath)
        for rect in facerect:
			x = rect[0] - 50
			y = rect[1] - 50
			width = rect[2] + 100
			height = rect[3] + 100
			dst = img[y : y + 10 + 3 * height, x : x + 30 + width]
			#imgResize = cv2.resize(dst,None,fx=dst.shape[0]/resize,fy=dst.shape[1]/resize)
			imgResize = cv2.resize(dst,(resize,resize))
			newImage_path = savePath + "/" + str(len(imageList) + i) + '.png'
			cv2.imwrite(newImage_path, imgResize)
			i = i + 1

	time.sleep(3)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='face detect')
	parser.add_argument('--detect', '-d', type=str, default='')
	parser.add_argument('--output', '-o', type=str, default='')
	parser.add_argument('--size', '-s', type=int, default=128)
	args = parser.parse_args()

    # detect face with already gathrede frame images
	#faceDetect(args.detect, args.output, args.size)

	# gather data for making model
	for i in range(camera):
		cap.append(cv2.VideoCapture(i))
	while True:
		ret, frame1 = cap[1].read()
		ret, frame2 = cap[2].read()
		frameResize1 = cv2.resize(frame1, (800, 800))
		frameResize2 = cv2.resize(frame2, (850, 850))
		cv2.imshow('camera1 capture', frameResize1)
		cv2.imshow('camera2 capture', frameResize2)
		gatherData(frameResize1, cameraNum[1], args.detect)
		gatherData(frameResize2, cameraNum[2], args.detect)

		k = cv2.waitKey(10)
		if k == 27:
			break

	# detect face after gathering frame images
	faceDetect(args.detect, args.output, args.size)

	cap[1].release()
	cap[2].release()
	cv2.destroyAllWindows()

	# gather data for predict
	'''
	for i in camera:
		cap.append(cv2.VideoCapture(i))
	while True:
		ret, frame0 = cap[0].read()
		ret, frame1 = cap[1].read()
		cv2.imshow('predicted data', frame0)
		cv2.imshow('predicted data', frame0)
		predictedData0 = Process(target = faceDetect4model(frame0, 0, args.detect, args.output, args.size),
		                         args = (frame0, cameraNum[0]))
		predictedData1 = Process(target = faceDetect4model(frame1, 0, args.detect, args.output, args.size),
		                         args = (frame1, cameraNum[1])

	    # multiprocessing
		predictedData0.start()
		predictedData0.join()

		predictedData1.start()
		predictedData1.join()

		# escape from the roop
		k = cv2.waitKey(10)
		if k == 27:
			break

	cap[0].release()
	cap[1].release()
	cv2.destroyAllWindows()
	'''
