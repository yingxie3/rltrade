import numpy as np 
import tensorflow as tf 
import datetime

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
        for current in range(start, end):
            currentWeekDay = self.dates[current].weekday()
            if currentWeekDay < lastWeekday:
                # a new week
                currentWeek += 1
                self.weeklyDates.append(self.dates[current])
                self.weeklyPrices[currentWeek][self.OPEN_INDEX] = self.prices[current][self.OPEN_INDEX]

            self.weeklyPrices[currentWeek][self.VOLUME_INDEX] += self.prices[current][self.VOLUME_INDEX]
            self.weeklyPrices[currentWeek][self.CLOSE_INDEX] = self.prices[current][self.CLOSE_INDEX]
            lastWeekday = currentWeekDay

    # get one data point for training, this includes the daily and weekly
    # open/close/volume
    def getOne():
        return None