from alexnet import Alex
from cnn_basic import CNN
from cnn_hoseiuniv import HoseiCNN
from nin import NIN
from googlenet import GoogLenet
from labeling import labeling

import numpy as np
import argparse
from datetime import datetime

import chainer
import chainer.functions as F
import chainer.links as L
import chainer.serializers
from chainer.datasets import tuple_dataset
from chainer import Chain, Variable, optimizers
from chainer import training
from chainer.training import extensions


def main():
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--batchsize', '-b', type=int, default=20)
    parser.add_argument('--epoch', '-e', type=int, default=20)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', default='result_basic')
    parser.add_argument('--channel', '-c', type=int, default=3)

    args = parser.parse_args()

    print('GPU: {}'.format(args.gpu))
    print('# Minbatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    # prepare datasets
    data = []
    data.append(np.asarray(['./datasets/concentration/', 0]))
    data.append(np.asarray(['./datasets/non_concentration/', 1]))
    data.append(np.asarray(['./datasets/others/', 2]))
    train, test = labeling(data, args.channel)

    model = L.Classifier(CNN())
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu()

    # setup the optimizer
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    #if args.model != '' and args.optimizer != '':
        #chainer.serializers.load_npz(args.model, model)
        #chainer.serializers.load_npz(args.optimizer, optimizer)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(
        test, args.batchsize, repeat=False, shuffle=False)

    # setup the trainer
    updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
    trainer = training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out)

    # evaluate the model with dataset for each epoch
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))

    # dump a computational graph from 'loss' variable
    trainer.extend(extensions.dump_graph('main/loss'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
            'main/accuracy', 'validation/main/accuracy']
    ))

    trainer.run()

    # save models
    output = "output" + str(len(data))
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    modelName = output + '_' + date + '.model'
    optimizerName = output + '_' + date + '.state'

    chainer.serializers.save_npz(args.out + "/" + modelName, model)
    chainer.serializers.save_npz(args.out + "/" + optimizerName, optimizer)


if __name__ == '__main__':
    main()
