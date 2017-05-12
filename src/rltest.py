import numpy as np 
import tensorflow as tf 
import datetime
import argparse
import pdb
import os
import pickle
import json
import pdb
from rltrain import Position
from rltrain import Company
from rltrain import createModel
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from keras.optimizers import adadelta

def test(stockName):
    currentDir = os.path.dirname(os.path.realpath(__file__))
    model, board = createModel()

    try:
        model.load_weights("model.h5")
    except OSError:
        print("can't find model file to load")

    filename = "{}/../data/train/{}.p".format(currentDir, stockName)
    company = pickle.load(open(filename, 'rb'))
    position = Position(company)

    done = False
    nextPriceDelta = position.getOne().reshape((-1, position.WIDTH, position.HEIGHT, 1))
    total = 1
    count = [0, 0, 0]
    while not done:
        priceDelta = nextPriceDelta
        q = model.predict(priceDelta)[0]
        action = np.argmax(q)
        position.holding = position.actionList[action]
        count[action] += 1

        reward, done = position.advance()
        nextPriceDelta = position.getOne().reshape((-1, position.WIDTH, position.HEIGHT, 1))
        total *= (1+reward)
        print("{} {:.4f} position {} reward {:.4f} total {:.4f}".format(position.company.dates[position.current],
            position.company.prices[position.current][Company.CLOSE_INDEX], position.holding, reward, total))
    
    print("total sell {} hold {} buy {}".format(count[0], count[1], count[2]))

def printModel():
    model, board = createModel()
    try:
        model.load_weights("model.h5")
    except OSError:
        print("can't find model file to load")

    w = model.get_weights()
    pdb.set_trace()

def main():
    parser = argparse.ArgumentParser(description='Parsing and training')
    parser.add_argument("-t", "--test", help="Test using the specified stock, can be all")
    parser.add_argument("-d", "--device", help="Which device to use, cpu or gpu")
    parser.add_argument("-p", "--print", help="Print out the weights", action="store_true")

    args = parser.parse_args()
    device = 'cpu' # use cpu by default
    
    # dump the data for one stock, for data verification only
    if args.device != None:
        device = args.device

    if args.test != None:
        test(args.test)
    elif args.print != None:
        printModel()

if __name__ == '__main__':
    main()
