import numpy as np 
import tensorflow as tf 
import datetime
import argparse
import pdb
import os
import pickle
import json
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from keras.optimizers import adadelta
from shutil import copyfile

# global parameters
EPOCH_PER_STOCK = 2 # the number of epochs per training run

# All the data related to one company
# Each quote is an array containing [date, open, close, volume]
# the quotes are stored in increasing order chronologically
class Company(object):
    DAY_INCREMENT = 0.25
    OPEN_INDEX = 0
    CLOSE_INDEX = 1
    VOLUME_INDEX = 2
    MAX_INDEX = VOLUME_INDEX

    def __init__(self, stockName):
        self.name = stockName

    def resetData(self, datalen):
        self.prices = []
        self.dates = []
        self.days = []

    def getDataLen(self):
        return len(self.dates)

    def addEntry(self, d, o, c, v):
        if len(self.dates) != 0:
            lastDate = self.dates[-1]
            lastPrice = self.prices[-1]

            # we always fill in the missing data, using the last valid data
            while (d - lastDate).days > 1:
                lastDate += datetime.timedelta(days=1)
                lastDay = lastDate.weekday()
                if lastDay >= 0 and lastDay <= 4:
                    self.dates.append(lastDate)
                    self.days.append(lastDay * self.DAY_INCREMENT)
                    self.prices.append(lastPrice)

        self.dates.append(d)
        self.days.append(d.weekday() * self.DAY_INCREMENT)
        self.prices.append([o, c, v])

    # we only generate data for complete weeks
    def generateWeeklyData(self):
        assert len(self.dates) == len(self.prices)
        days = (self.dates[-1] - self.dates[0]).days
        self.prices = np.array(self.prices)

        weeks = int(days / 7)
        self.weeklyDates = []
        self.weeklyPrices = np.zeros((weeks+1, self.MAX_INDEX+1))

        # find starting day (always a Monday) and ending day (always a Friday)
        start = 0
        for start in range(len(self.dates)):
            if self.dates[start].weekday() == 0:
                break

        end = len(self.dates) - 1
        while True:
            if self.dates[end].weekday() == 4:
                break
            end -= 1

        # init the value to make the loop easy
        lastWeekday = 4
        currentWeek = -1
        for current in range(start, end+1):
            currentWeekDay = self.dates[current].weekday()
            if currentWeekDay < lastWeekday:
                # a new week
                currentWeek += 1
                self.weeklyDates.append(self.dates[current])
                self.weeklyPrices[currentWeek][self.OPEN_INDEX] = self.prices[current][self.OPEN_INDEX]

            self.weeklyPrices[currentWeek][self.VOLUME_INDEX] += self.prices[current][self.VOLUME_INDEX]
            self.weeklyPrices[currentWeek][self.CLOSE_INDEX] = self.prices[current][self.CLOSE_INDEX]
            lastWeekday = currentWeekDay

# A position is the company, plus the holding information. 
class Position(object):
    WIDTH = 50
    HEIGHT = 5 # just weekday indicator, open/close and weekly open/close, no volume, this must match with getOne slice
    #actionList = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    actionList = [-1.0, 0.0, 1.0]
    ACTION_SIZE = len(actionList)
    ACTION_INCREMENT = 2.0/(ACTION_SIZE-1)

    def reset(self):
        self.currentWeek = self.WIDTH # index to the current week, this allows WIDTH/(WIDTH-1)
        # move to the next week, this is the date we will match for the daily data.
        weekDate = self.company.weeklyDates[self.WIDTH+1]

        # index for the current day. The date above is the Monday's date for the weekly data.
        # The current need to be at least that. The following formula guarantees it.
        self.current = (self.WIDTH+1) * 5 + 5
        date = self.company.dates[self.current]
        assert date >= weekDate

        # move date back to be the same as weekDate
        while date > weekDate:
            self.current -= 1
            date = self.company.dates[self.current]
        assert self.current >= (self.WIDTH+1) * 5

        # we always store the np array representation of the current data.
        self.days = np.array(self.company.days[self.current+1-self.WIDTH:self.current+1]).reshape((-1, 1))
        self.dailyPriceDelta = self.company.prices[self.current+1-self.WIDTH:self.current+1] /       \
            self.company.prices[self.current-self.WIDTH:self.current] - 1.0
        self.weeklyPriceDelta = self.company.weeklyPrices[1:self.WIDTH+1] / self.company.weeklyPrices[0:self.WIDTH] - 1.0
        #self.data = np.concatenate((dailyPriceDelta, weeklyPriceDelta), axis=1)

    def __init__(self, comp):
        self.holding = 0 # 0.2 means 20% cash in this stock
        self.company = comp
        self.reset()

    # advance current by one trading day, returns the immediate incremental reward (in percentage)
    # and whether this is the last entry.
    def advance(self):
        previousDay = self.company.days[self.current]
        self.current += 1
        currentDay = self.company.days[self.current]
        self.days = np.concatenate((self.days[1:self.WIDTH], [[currentDay]]), axis=0)

        if currentDay < previousDay:
            # a new week, advance week array
            self.currentWeek += 1
            newDelta = self.company.weeklyPrices[self.currentWeek] / self.company.weeklyPrices[self.currentWeek-1] - 1.0
            self.weeklyPriceDelta = np.concatenate((self.weeklyPriceDelta[1:self.WIDTH], [newDelta]), axis=0)
        
        assert self.company.dates[self.current] > self.company.weeklyDates[self.currentWeek]
        newDelta = self.company.prices[self.current] / self.company.prices[self.current-1] - 1.0
        self.dailyPriceDelta = np.concatenate((self.dailyPriceDelta[1:self.WIDTH], [newDelta]), axis=0)

        # get the reward using close price
        done = self.current+1 == len(self.company.dates)
        #done = self.current >= 1000
        return newDelta[self.company.CLOSE_INDEX] * self.holding, done

    # get the current state (50x6 graph)
    def getOne(self):
        return np.concatenate((self.days, self.dailyPriceDelta[:,0:2], self.weeklyPriceDelta[:,0:2]), axis=1)

# When playing, we go through the Position in time order, and store the result below. This allows
# us to use random batches when training.
class ReplayHistory(object):
    def __init__(self, maxMemory=1000, discount=.999):
        self.maxMemory = maxMemory
        self.memory = []
        self.discount = discount

    def remember(self, priceDelta, holding, reward, newPriceDelta, isLast):
        self.memory.append([[priceDelta, holding, reward, newPriceDelta], isLast])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]
    
    # get batchSize of training samples, mapping states (?x50x6) to action reward values (?x11)
    def getBatch(self, model, actionSize, batchSize, printTarget):
        count = min(len(self.memory), batchSize)
        inputs = np.zeros((count, Position.WIDTH, Position.HEIGHT, 1))
        targets = np.zeros((count, actionSize))

        for i, idx in enumerate(np.random.randint(0, len(self.memory), size=count)):
            priceDelta, holding, reward, newPriceDelta = self.memory[idx][0]
            isLast = self.memory[idx][1] or newPriceDelta[0][-1][0][0] == 1.0 # close out position at Friday.
            holdingIndex = int((holding + 1) / Position.ACTION_INCREMENT)

            inputs[i] = priceDelta
            targets[i]= model.predict(priceDelta)[0] # target value is the action's future value
            if not isLast:
                # calculate Q value from the new state.
                Q = np.max(model.predict(newPriceDelta)[0])
            else:
                Q = 1.0

            # this is the compounding model, the method we use during testing must match this.
            newTarget = (1 + reward) * Q * self.discount
            if printTarget:
                print("target for action {} index {}: {} => {} = (1 + {}) * {} * {}".format(holding, holdingIndex, 
                    targets[i][holdingIndex], newTarget, reward, Q, self.discount))
            targets[i][holdingIndex] = newTarget
        
        return inputs, targets

# Implements the neural network model.
def createModel():
    # parameters
    num_actions = Position.ACTION_SIZE
    batch_size = 50
    grid_size = 10

    model = Sequential()
    model.add(Conv2D(64, (Position.HEIGHT, Position.HEIGHT), input_shape=(Position.WIDTH, Position.HEIGHT, 1), 
        strides=(1, 1), padding='valid', name='conv1', activation='relu'))
    model.add(Conv2D(64, (3, 1), strides=(1, 1), padding='valid', name='conv2', activation='relu'))

    '''
    model.add(Conv2D(128, (3, 1), strides=(1, 1), padding='valid', name='conv3', activation='relu'))
    model.add(Conv2D(128, (3, 1), strides=(1, 1), padding='valid', name='conv4', activation='relu'))
    '''

    model.add(Flatten(name='flatten'))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_actions))

    model.compile(loss='mse', optimizer=adam())

    board = TensorBoard(log_dir='./logs', histogram_freq=2, write_graph=True, write_images=False)
    board.set_model(model)
    return model, board

def dump(filename):
    if filename == None:
        print("you must specify the data file you want to dump through --input")
        return
    company = pickle.load(open(filename, 'rb'))

    print("Dumping data for {}".format(company.name))
    print("============ Daily data =============")
    for i in range(0, len(company.dates)):
        print("{} open {} close {} volume {}".format(company.dates[i], company.prices[i][company.OPEN_INDEX], 
            company.prices[i][company.CLOSE_INDEX], company.prices[i][company.VOLUME_INDEX]))
    print("============ Weekly data =============")
    for i in range(0, len(company.weeklyDates)):
        print("{} open {} close {} volume {}".format(company.weeklyDates[i], company.weeklyPrices[i][company.OPEN_INDEX], 
            company.weeklyPrices[i][company.CLOSE_INDEX], company.weeklyPrices[i][company.VOLUME_INDEX]))


def play(filename):
    #pdb.set_trace()
    if filename == None:
        print("you must specify the data file you want to dump through --input")
        return
    company = pickle.load(open(filename, 'rb'))
    position = Position(company)
    history = ReplayHistory(discount=1.0)

    print("Playing history for {}".format(company.name))
    position.holding = 1.0
    nextPriceDelta = position.getOne()
    done = False
    while not done:
        priceDelta = nextPriceDelta
        holding = position.holding
        reward, done = position.advance()
        nextPriceDelta = position.getOne()

        history.remember(priceDelta, holding, reward, nextPriceDelta, done)

    total = 1.0
    count = 0
    for idx in range(len(history.memory)):
        priceDelta, holding, reward, newPriceDelta = history.memory[idx][0]
        isLast = history.memory[idx][1]
        holdingIndex = (holding + 1) / Position.ACTION_INCREMENT
        total *= (1 + reward)
        print("reward {} total {}".format(reward, total))

        if count < 5:
            count += 1
            for w in range(position.WIDTH):
                print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ".format(priceDelta[w][0], priceDelta[w][1], priceDelta[w][2], 
                    priceDelta[w][3], priceDelta[w][4]))
            print("")

def train(stockName):
    currentDir = os.path.dirname(os.path.realpath(__file__))
    epsilon = 0.1  # exploration
    batchSize = 100
    validationData = [np.ones((batchSize, Position.WIDTH, Position.HEIGHT, 1))]
    model, board = createModel()
    history = ReplayHistory(discount=1.0)

    try:
        model.load_weights("model.h5")
    except OSError:
        print("can't find model file to load")

    stockList = []
    if stockName == 'ALL':
        for filename in os.listdir("{}/../data/train".format(currentDir)):
            if filename.endswith(".p"):
                filear = filename.split('.')
                stockList.append(filear[0])
    else:
        stockList.append(stockName)

    for epoch in range(len(stockList)*EPOCH_PER_STOCK):
        if epoch % EPOCH_PER_STOCK == 0:
            stockName = stockList[int(epoch / EPOCH_PER_STOCK)]
            print("=====training {}=====".format(stockName))
            filename = "{}/../data/train/{}.p".format(currentDir, stockName)
            company = pickle.load(open(filename, 'rb'))
            position = Position(company)

        # run the company from beginning to end in each epoch
        position.holding = 0.0
        position.reset()

        nextPriceDelta = position.getOne().reshape((-1, position.WIDTH, position.HEIGHT, 1))
        done = False
        while not done:
            priceDelta = nextPriceDelta

            # with some probability we take a random holding position
            if np.random.rand() <= epsilon:
                position.holding = position.actionList[np.random.randint(0, position.ACTION_SIZE)]
            else:
                q = model.predict(priceDelta)[0]
                action = np.argmax(q)
                position.holding = position.actionList[action]

            reward, done = position.advance()
            nextPriceDelta = position.getOne().reshape((-1, position.WIDTH, position.HEIGHT, 1))

            history.remember(priceDelta, position.holding, reward, nextPriceDelta, done)

            # Now get a batch from history and train
            inputs, targets = history.getBatch(model, position.ACTION_SIZE, 32, position.current % 100 == 0)
            loss = model.train_on_batch(inputs, targets)

            if position.current % 100 == 0:
                print("day {} week {} holding {} reward {} | loss {}".format(position.current, position.currentWeek, position.holding, reward, loss))
                for w in range(10):
                    print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ".format(priceDelta[0][w][0][0], priceDelta[0][w][1][0],
                        priceDelta[0][w][2][0], priceDelta[0][w][3][0], priceDelta[0][w][4][0]))
                print("")

        # do some printing and model saving after one epoch
        print("=========== Epoch {:03d} | Loss {:.4f}".format(epoch, loss))

        # on_epoch_end requires validataData to be present. It doesn't really need
        # the data to get the histogram, so we just give is a pre-fabricated one.
        board.validation_data = validationData
        logs = {'loss': loss}
        board.on_epoch_end(epoch, logs)

        # Save trained model weights and architecture, this will be used by the visualization code
        print("Saving model")
        try:
            copyfile('model.h5', 'model.h5.saved')
            copyfile('model.json', 'model.json.saved')
        except FileNotFoundError:
            print("didn't find model files to save")

        model.save_weights("model.h5", overwrite=True)
        with open("model.json", "w") as outfile:
            json.dump(model.to_json(), outfile)

def main():
    parser = argparse.ArgumentParser(description='Parsing and training')
    parser.add_argument("-d", "--dump", help="Dump data only.")
    parser.add_argument("-p", "--play", help="Play through one stock history contained in the specified file.")
    parser.add_argument("-t", "--train", help="Train using the specified stock, can be all")

    args = parser.parse_args()

    # dump the data for one stock, for data verification only
    if args.dump != None:
        dump(args.dump)
    elif args.play != None:
        play(args.play)
    elif args.train != None:
        train(args.train)

if __name__ == '__main__':
    main()