import numpy as np 
import tensorflow as tf 
import datetime
import argparse
import pdb
import os
import pickle

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
class Position(Company):
    WIDTH = 50
    HEIGHT = 6 
    actionList = [-1.0, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ACTION_SIZE = len(actionList)

    def __init__(self):
        self.holding = 0 # 0.2 means 20% cash in this stock
        self.currentWeek = self.WIDTH # index to the current week, this allows WIDTH/(WIDTH-1)

        # move to the next week, this is the date we will match for the daily data.
        weekDate = self.weeklyDates[self.WIDTH+1]

        # index for the current day. The date above is the Monday's date for the weekly data.
        # The current need to be at least that. The following formula guarantees it,
        # because some of the weeks have fewer than 5 days.
        self.current = (self.WIDTH+1) * 5
        date = self.dates[self.current]
        assert date >= weekDate

        # move date back to be the same as weekDate
        while date > weekDate:
            self.current -= 1
            date = self.dates[self.current]
        assert self.current >= (self.WIDTH+1) * 5 - 10  # only that many holidays in a year

        # we always store the np array representation of the current data.
        self.dailyPriceDelta = self.prices[self.current+1-self.WIDTH:self.current+1] /       \
            self.prices[self.current-self.WIDTH:self.current] - 1.0
        self.weeklyPriceDelta = self.weeklyPrices[1:self.WIDTH+1] / self.weeklyPrices[0:self.WIDTH] - 1.0
        #self.data = np.concatenate((dailyPriceDelta, weeklyPriceDelta), axis=1)

    # advance current by one trading day, returns the immediate reward (in percentage)
    def advance(self):
        previousDay = self.dates[self.current].weekday()
        self.current += 1

        if self.dates[self.current].weekday() < previousDay:
            # a new week, advance week array
            self.currentWeek += 1
            np.roll(self.weeklyPriceDelta, 1, axis=0)
            self.weeklyPriceDelta[self.WIDTH-1] = self.weeklyPrices[self.currentWeek] / self.weeklyPrices[self.currentWeek-1] - 1.0
        
        assert self.dates[self.current] > self.weeklyDates[self.currentWeek]
        np.roll(self.dailyPriceDelta, 1, axis=0)
        self.dailyPriceDelta[self.WIDTH-1] = self.prices[self.current] / self.prices[self.current-1] - 1.0

        # get the reward using close price
        return self.dailyPriceDelta[self.WIDTH-1] * self.holding

    # given holding and index, get the reward for the next day
    def getReward(self, dailyIndex, holding):
        return (self.prices[dailyIndex+1] / self.prices[dailyIndex] - 1.0) * holding

    # get the current state (50x6 graph)
    def getOne(self):
        return np.concatenate((self.dailyPriceDelta, self.weeklyPriceDelta), axis=1)

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
    def getBatch(self, model, position, actionSize, batchSize):
        count = min(len(self.maxMemory), batchSize)
        inputs = np.zeros((count, Position.WIDTH, Position.HEIGHT, 1))
        targets = np.zeros((count, actionSize))

        for i, idx in enumerate(np.random.randint(0, len(self.maxMemory), size=count)):
            priceDelta, holding, reward, newPriceDelta = self.memory[idx][0]
            isLast = self.memory[idx][1]
            holdingIndex = (holding + 1) / 0.2

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

def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='Parsing and training')
    parser.add_argument("-d", "--dump", help="Dump data only.")

    args = parser.parse_args()

    # dump the data for one stock, for data verification only
    if args.dump != None:
        dump(args.dump)

if __name__ == '__main__':
    main()