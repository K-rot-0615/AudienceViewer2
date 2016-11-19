import pylab
import argparse
import json


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--logFile', '-l', type=str, default='./result/log')
    args = parser.parse_args()

    x = []
    y = []

    for log in open(args.logFile):
        logFile = json.loads(log)
        x.append(logFile["epoch"])
        y.append(logFile["main/accuracy"])

    pylab.xlabel("epoch")
    pylab.ylabel("main/accuracy")
    pylab.plot(x,y)
    pylab.show()
