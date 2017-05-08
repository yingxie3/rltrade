import numpy as np 
import argparse
import rltrain
import dateutil
import json
import os
import pdb
import pickle
import urllib.request

BASEURL="https://www.quandl.com/api/v3/datatables/WIKI/PRICES.json?qopts.columns=date,adj_open,adj_close,adj_volume"

# process the Json file. Returns a rltrain.Company object
def processJsonData(companyName, filename):
    # The index as presented in json files
    DATE_INDEX = 0
    OPEN_INDEX = 1
    CLOSE_INDEX = 2
    VOLUME_INDEX = 3
    MAX_INDEX = VOLUME_INDEX

    if not filename.endswith(".json"):
        return
    
    print("\nprocessing {}".format(filename))

    with open(filename) as jsonFile:
        jsonData = json.load(jsonFile)
        dataArray = jsonData['datatable']['data']
        company = rltrain.Company(companyName)

        # clean the data
        if jsonData == None:
            print("discarding stock")
            return None

        company.resetData(len(dataArray))
        for index, d in enumerate(dataArray):
            #sanity check
            for i in range(MAX_INDEX):
                if d[i] == None or d[i] == 0:
                    print("found zero entry: {}".format(d))
                    # we discard all data before this
                    company.resetData(len(dataArray) - (index+1))

            dateValue = dateutil.parser.parse(d[DATE_INDEX])
            company.addEntry(dateValue, d[OPEN_INDEX], d[CLOSE_INDEX], d[VOLUME_INDEX])

        # we only allow companies with enough data.
        if company.getDataLen() < 500:
            print("discarding stock {}".format(companyName))
            return None
        return company
    return None


# we download stock data from quandl if needed and save in the data/downloaded directory,
# then we process them into the processed directory.
def main():
    currentDir = os.path.dirname(os.path.realpath(__file__))

    parser = argparse.ArgumentParser(description='downloading data from quandl')
    parser.add_argument("-i", "--input", help="Set input file name.")
    parser.add_argument("-o", "--output", help="Set output directory name.")
    parser.add_argument("-t", "--train", help="Set training directory name.")
    parser.add_argument("-k", "--key", help="Set the quandl api key")

    args = parser.parse_args()
    if args.input == None:
        args.input = "{}/../data/sp500.csv".format(currentDir)
    if args.output == None:
        args.output = "{}/../data/downloaded".format(currentDir)
    if args.train == None:
        args.train = "{}/../data/train".format(currentDir)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    if not os.path.exists(args.train):
        os.makedirs(args.train)

    # download again if the output directory doesn't have any json files
    companies = []
    for filename in os.listdir(args.output):
        if filename.endswith(".json"): 
            companies.append(filename.split('.')[0])    

    if len(companies) == 0:
        if args.key == None:
            print("argements not specified, run with -h to see the help")
            exit(0)

        with open(args.input, 'r') as fin:
            fin.readline() # skip the first line, which is the header
            for line in fin.readlines():
                ar = line.split(',')

                print("reading {}\n".format(ar[0]))
                urlstr = "{}&api_key={}&ticker={}".format(BASEURL, args.key, ar[0])
                urllib.request.urlretrieve(urlstr, "{}/{}.json".format(args.output, ar[0]))
                companies.append(ar[0])

    # Process the json files into binary files. We currently do this in a single thread. 
    # This is OK since we only do this once. No need to be fancy.
    for cName in companies:
        company = processJsonData(cName, "{}/{}.json".format(args.output, cName))
        if company != None:
            company.generateWeeklyData()
            pickle.dump(company, open("{}/{}.p".format(args.train, company.name), "wb"))

if __name__ == '__main__':
    main()