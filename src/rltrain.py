import numpy as np 
import tensorflow as tf 
import datetime

# All the data related to one company
# Each quote is an array containing [date, open, close, volume]
# the quotes are stored in increasing order chronologically
class Company(object):
    def __init__(self, stockName):
        self.name = stockName
        self.data = []

    def addEntry(self, d, o, c, v):
        self.data.append([d, o, c, v])

    