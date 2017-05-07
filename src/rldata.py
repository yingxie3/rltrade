import numpy as np 
import argparse
import rltrain
import dateutil
import json
import os
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

        for d in dataArray:
            #sanity check
            for i in range(MAX_INDEX):
                if d[i] == None or d[i] == 0:
                    print("found zero entry: {}".format(d))
                    # we discard all data before this
                    company.data = []

            dateValue = dateutil.parser.parse(d[DATE_INDEX])
            company.addEntry(dateValue, d[OPEN_INDEX], d[CLOSE_INDEX], d[VOLUME_INDEX])

        # we only allow companies with enough data.
        if len(company.data) < 500:
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
    parser.add_argument("-k", "--key", help="Set the quandl api key")

    args = parser.parse_args()
    if args.input == None:
        args.input = "{}/../data/sp500.csv".format(currentDir)
    if args.output == None:
        args.output = "{}/../data/downloaded".format(currentDir)

    # download again if the output directory doesn't have any json files
    found = False
    for filename in os.listdir(args.output):
        if filename.endswith(".json"): 
            found = True
            break

    if not found:
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

    # Process the json files into binary files. We currently do this in a single thread. 
    # This is OK since we only do this once. No need to be fancy.

if __name__ == '__main__':
    main()