import csv
import numpy as np 
from math import sqrt
from math import pi
from math import exp
import sys
from statistics import mean

def convert(x):
    try:
        return float(x)
    except ValueError:
        if x == 'yes':
            return 1
        else: 
            return 0
        

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
def separate_by_class(dataset):
	separated = dict()
	for i in range(len(dataset)):
		vector = dataset[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated

def summarize_by_class(dataset):
	separated = separate_by_class(dataset)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_dataset(rows)
	return summaries

def summarize_dataset(dataset):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
	del(summaries[-1])
	return summaries

def mean(numbers):
	return sum(numbers)/float(len(numbers))

def pdf(x, mean, stdev):
	exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
	return (1 / (sqrt(2 * pi) * stdev)) * exponent

def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)

def NB(training_input,tesing_input):
    mean = mean(training_input)
    sd = stdev(training_input,mean)
    #have to figure out what to do next
    no_yes = 'asa'
    return 1

    
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
   
    if algorithm == 'NB':
       result = NB(training_input,tesing_input)
    elif 'NN' in algorithm:
       k = int(algorithm.strip("NN"))
       for i in tesing_input:
           result.append(KNN(k,training_data,tesing_input))

