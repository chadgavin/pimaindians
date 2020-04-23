import csv
import numpy as np 
import math
import sys

def convert(x):
    try:
        return float(x)
    except ValueError:
        return x.title()

def extract(input_file):
    data = []
    lines = []
    data_type =[]
    with open(input_file,'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for i in lines:
        data.append(i.split(","))
    data = [[convert(element) for element in entry] for entry in data]

    return data
# def NB(training_input,tesing_input):
    
# def KNN(k,training_data,tesing_input):
   
def main(argv):
	print('a')

if __name__ == "__main__":
    results =[]
    training_data = sys.argv[1]
    testing_data = sys.argv[2]
    algorithm = sys.argv[3]
    training_input = extract(training_data)
    tesing_input = extract(testing_data)
    
    print(training_input)
    if algorithm == 'NB':
       result = NB(training_input,tesing_input)
    elif 'NN' in algorithm:
       k = int(algorithm.strip("NN"))
       for i in tesing_input:
           result.append(KNN(k,training_data,tesing_input))

