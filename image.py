# -*- coding:utf-8 -*-

import argparse
import cv2
import time
import os
import os.path
import glob
from PIL import Image
from datetime import datetime

camera = 1
cameraNum = [0, 1, 2, 3]
x = []
y = []
width = []
height = []


def dataRead(path):
	data = []
	imgList = glob.glob(path + '*')
	for imgName in imgList:
		data.append(imgName)
	return data


def faceDetect(frame, cameraNum, pre_detectPath, savePath, resize):

	cascade_path = "./lib/haarcascade_frontalface_default.xml"
	cascade = cv2.CascadeClassifier(cascade_path)

	now = datetime.now().strftime('%Y%m%d%H%M%S')
	now_image = pre_detectPath + now + '.png'
	if os.path.isdir(pre_detectPath) == False:
		os.mkdir(pre_detectPath)
		print 'make pre_detectPath!'
	cv2.imwrite(now_image, frame)

	img = cv2.imread(now_image)
	#frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
	facerect = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
	if len(facerect) > 0:
		print 'Face has been detected'
		if os.path.isdir(savePath) == False:
			os.mkdir(savePath)
			print 'make image path'
		else:
			print 'path already exists'

	i = 1
	imageList = glob.glob(savePath + "*")
	for rect in facerect:
		x.append(rect[0] - 50)
		y.append(rect[1] - 50)
		width.append(rect[2] + 100)
		height.append(rect[3] + 100)
		for n in range(len(x)):
			dst = img[y[n]:y[n] + 10 + 3 * height[n], x[n]:x[n] + 30 + width[n]]
			imgResize = cv2.resize(dst,(resize,resize))
			newImage_path = savePath + str(len(imageList) + i) + '.png'
			#while True:
			    #if os.path.isfile(newImage_path):
				    #i += 1
                #else:
				    #cv2.imwrite(newImage_path, imgResize)
				    #break
			cv2.imwrite(newImage_path, imgResize)


def faceDetect4Predict(frame, cameraNum, pre_detectPath, savePath, resize):

	cascade_path = "./lib/haarcascade_frontalface_default.xml"
	cascade = cv2.CascadeClassifier(cascade_path)

	now = datetime.now().strftime('%Y%m%d%H%M%S')
	now_image = pre_detectPath + now + '.png'
	if os.path.isdir(pre_detectPath) == False:
		os.mkdir(pre_detectPath)
		print 'make pre_detectPath!'
	cv2.imwrite(now_image, frame)

	images = dataRead(pre_detectPath)
	for image in images:
		img = cv2.imread(image)
	    #frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	    #facerect = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
		facerect = cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=3, minSize=(10, 10))
		if len(facerect) > 0:
		    print 'Face has been detected'
		    if os.path.isdir(savePath) == False:
			    os.mkdir(savePath)
			    print 'make image path'
		    else:
			    print 'path already exists'

        i = 1
        imageList = glob.glob(savePath + "*")
        for rect in facerect:
			print rect[0]
			x.append(rect[0] - 50)
			y.append(rect[1] - 50)
			width.append(rect[2] + 100)
			height.append(rect[3] + 100)
			for n in range(len(x)):
				dst = img[y[n]:y[n] + 10 + 3 * height[n], x[n]:x[n] + 30 + width[n]]
			    #imgResize = cv2.resize(dst,None,fx=dst.shape[0]/resize,fy=dst.shape[1]/resize)
				imgResize = cv2.resize(dst,(resize,resize))
				newImage_path = savePath + "/" + str(len(imageList) + i) + '.png'
			    #while True:
			        #if os.path.isfile(newImage_path):
				        #i += 1
                    #else:
				        #cv2.imwrite(newImage_path, imgResize)
				        #break
				cv2.imwrite(newImage_path, imgResize)
				i = i + 1

			time.sleep(3)


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='face detect')
	parser.add_argument('--detect', '-d', type=str, default='')
	parser.add_argument('--output', '-o', type=str, default='')
	parser.add_argument('--size', '-s', type=int, default=128)
	args = parser.parse_args()

	# gather data for making model
	'''
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		cv2.imshow('camera capture', frame)
		pre_predictData = faceDetect(frame, cameraNum[0], args.detect, args.output, args.size)

		k = cv2.waitKey(10)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
	'''

	# gather data for predict
	cap = cv2.VideoCapture(0)
	while True:
		ret, frame = cap.read()
		cv2.imshow('predicted data', frame)
		predictedData = faceDetect4Predict(frame, 0, args.detect, args.output, args.size)

		# escape from the roop
		k = cv2.waitKey(10)
		if k == 27:
			break

	cap.release()
	cv2.destroyAllWindows()
