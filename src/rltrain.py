import numpy as np 
import tensorflow as tf 
import datetime
import argparse
import pdb
import os
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Flatten
from keras.layers.convolutional import Conv2D
from keras.callbacks import TensorBoard
from keras.optimizers import adam
from keras.optimizers import adadelta

# All the data related to one company
# Each quote is an array containing [date, open, close, volume]
# the quotes are stored in increasing order chronologically
class Company(object):
    OPEN_INDEX = 0
    CLOSE_INDEX = 1
    VOLUME_INDEX = 2
    MAX_INDEX = VOLUME_INDEX

    def __init__(self, stockName):
        self.name = stockName

    def resetData(self, datalen):
        self.prices = np.zeros((datalen, self.MAX_INDEX+1))
        self.dates = []

    def getDataLen(self):
        return len(self.dates)

    def addEntry(self, d, o, c, v):
        count = len(self.dates)
        self.dates.append(d)
        self.prices[count] = [o, c, v]

    # we only generate data for complete weeks
    def generateWeeklyData(self):
        assert len(self.dates) == len(self.prices)
        days = (self.dates[len(self.dates)-1] - self.dates[0]).days
        weeks = int(days / 7)
        self.weeklyDates = []
        self.weeklyPrices = np.zeros((weeks, self.MAX_INDEX+1))

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
    HEIGHT = 6 
    actionList = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ACTION_SIZE = len(actionList)

    def reset(self):
        self.currentWeek = self.WIDTH # index to the current week, this allows WIDTH/(WIDTH-1)
        # move to the next week, this is the date we will match for the daily data.
        weekDate = self.company.weeklyDates[self.WIDTH+1]

        # index for the current day. The date above is the Monday's date for the weekly data.
        # The current need to be at least that. The following formula guarantees it,
        # because some of the weeks have fewer than 5 days.
        self.current = (self.WIDTH+1) * 5
        date = self.company.dates[self.current]
        assert date >= weekDate

        # move date back to be the same as weekDate
        while date > weekDate:
            self.current -= 1
            date = self.company.dates[self.current]
        assert self.current >= (self.WIDTH+1) * 5 - 10  # only that many holidays in a year

        # we always store the np array representation of the current data.
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
        previousDay = self.company.dates[self.current].weekday()
        self.current += 1

        if self.company.dates[self.current].weekday() < previousDay:
            # a new week, advance week array
            self.currentWeek += 1
            newDelta = self.company.weeklyPrices[self.currentWeek] / self.company.weeklyPrices[self.currentWeek-1] - 1.0
            self.weeklyPriceDelta = np.concatenate((self.weeklyPriceDelta[1:self.WIDTH], [newDelta]), axis=0)
        
        assert self.company.dates[self.current] > self.company.weeklyDates[self.currentWeek]
        newDelta = self.company.prices[self.current] / self.company.prices[self.current-1] - 1.0
        self.dailyPriceDelta = np.concatenate((self.dailyPriceDelta[1:self.WIDTH], [newDelta]), axis=0)

        # get the reward using close price
        return self.dailyPriceDelta[self.WIDTH-1][self.company.CLOSE_INDEX] * self.holding, self.current+1 == len(self.company.dates)

    # given holding and index, get the reward for the next day
    def getReward(self, dailyIndex, holding):
        return (self.company.prices[dailyIndex+1] / self.company.prices[dailyIndex] - 1.0) * holding

    # get the current state (50x6 graph)
    def getOne(self):
        return np.concatenate((self.dailyPriceDelta, self.weeklyPriceDelta), axis=1)

# When playing, we go through the Position in time order, and store the result below. This allows
# us to use random batches when training.
class ReplayHistory(object):
    def __init__(self, maxMemory=100000, discount=.999):
        self.maxMemory = maxMemory
        self.memory = []
        self.discount = discount

    def remember(self, priceDelta, holding, reward, newPriceDelta, isLast):
        self.memory.append([[priceDelta, holding, reward, newPriceDelta], isLast])
        if len(self.memory) > self.maxMemory:
            del self.memory[0]
    
    # get batchSize of training samples, mapping states (?x50x6) to action reward values (?x11)
    def getBatch(self, model, actionSize, batchSize):
        count = min(len(self.memory), batchSize)
        inputs = np.zeros((count, Position.WIDTH, Position.HEIGHT, 1))
        targets = np.zeros((count, actionSize))

        for i, idx in enumerate(np.random.randint(0, len(self.memory), size=count)):
            priceDelta, holding, reward, newPriceDelta = self.memory[idx][0]
            isLast = self.memory[idx][1]
            holdingIndex = int((holding + 1) / 0.2)

            inputs[i] = priceDelta
            targets[i]= model.predict(priceDelta)[0] # target value is the action's future value
            if not isLast:
                # calculate Q value from the new state.
                Q = np.max(model.predict(newPriceDelta)[0])
            else:
                Q = 1.0

            # this is the compounding model, the method we use during testing must match this.
            targets[i][holdingIndex] = (1 + reward) * Q * self.discount
        
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

    '''
    model.add(Conv2D(64, (3, 1), strides=(1, 1), padding='valid', name='conv2', activation='relu'))
    model.add(Conv2D(128, (3, 1), strides=(1, 1), padding='valid', name='conv3', activation='relu'))
    model.add(Conv2D(128, (3, 1), strides=(1, 1), padding='valid', name='conv4', activation='relu'))
    '''

    model.add(Flatten(name='flatten'))
    # model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(num_actions))

    model.compile(adam(lr=.001), "mse")

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
        holdingIndex = (holding + 1) / 0.2
        total *= (1 + reward)
        print("reward {} total {}".format(reward, total))

        if count < 5:
            count += 1
            for w in range(position.WIDTH):
                print("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} ".format(priceDelta[w][0], priceDelta[w][1], priceDelta[w][2], 
                    priceDelta[w][3], priceDelta[w][4], priceDelta[w][5]))
            print("")

def train(stockName):
    currentDir = os.path.dirname(os.path.realpath(__file__))
    epsilon = 0.05  # exploration
    batchSize = 100
    validationData = [np.ones((batchSize, Position.WIDTH, Position.HEIGHT, 1))]
    model, board = createModel()

    # model.load_weights("model.h5")
    #pdb.set_trace()
    filename = "{}/../data/train/{}.p".format(currentDir, stockName)
    company = pickle.load(open(filename, 'rb'))
    position = Position(company)
    history = ReplayHistory(discount=0.999)

    for epoch in range(1000):
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
            inputs, targets = history.getBatch(model, position.ACTION_SIZE, 32)
            loss = model.train_on_batch(inputs, targets)

            if epoch % 10 == 0:
                print("Epoch {:03d} | Loss {:.4f}".format(epoch, loss))

                # on_epoch_end requires validataData to be present. It doesn't really need
                # the data to get the histogram, so we just give is a pre-fabricated one.
                board.validation_data = validationData
                logs = {'loss': loss}
                board.on_epoch_end(epoch, logs)

            # Save trained model weights and architecture, this will be used by the visualization code
            if epoch % 100 == 0:
                print("Saving model")
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