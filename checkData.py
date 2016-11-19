import glob
import argparse
import fnmatch
from image import dataRead
from PIL import Image


# check whether the file ends png or not
def checkFormat(path):
    checkData = dataRead(path)
    for data in checkData:
        if fnmatch.fnmatch(data, '*.png'):
            continue
        else:
            print data


# check whether the file size is 128*128 or nor
def checkSize(path):
    checkData = glob.glob(path + '*')
    for data in checkData:
        img = Image.open(data)
        width = img.size[0]
        height = img.size[1]
        if width != 128 or height != 128:
            print data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', type=str, default='')
    args = parser.parse_args()

    if args.data != "":
        check1 = checkFormat(args.data)
        check2 = checkSize(args.data)
