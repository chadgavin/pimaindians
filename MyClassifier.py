import csv
import numpy as np 
# from math import sqrt
# from math import pi
# from math import exp
import math
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

def euclidean_distance(point1 , point2):
    squared_distance_sum = 0
    for i in range(len(point1)):
        squared_distance_sum += math.pow(point1[i] - point2[i], 2)
    dist = math.sqrt(squared_distance_sum)
    return dist

    
def KNN(k,training_data,testing_input):

    neighbours= []
    yes_count = 0
    no_count = 0

    for training_example in training_data:
        point1 = training_example[:-1]
        point2 = testing_input
        distance_result = euclidean_distance(point1,point2)
        neighbours.append((training_example[-1],distance_result))

    neighbours.sort(key=lambda neighbour: neighbour[1])

    for i in range(k):
        if neighbours[i][0] == 1:
            yes_count += 1
        elif neighbours[i][0] == 0:
            no_count += 1


    if yes_count > no_count:
        return "yes"
    elif no_count > yes_count:
        return "no"
    elif yes_count == no_count:
        return "yes"


def main(argv):
	print('a')

if __name__ == "__main__":
    results =[]
    training_data = sys.argv[3]
    testing_data = sys.argv[4]
    algorithm = sys.argv[5]
    training_input = extract(training_data)
    testing_input = extract(testing_data)


    if algorithm == 'NB':
       result = NB(training_input,tesing_input)
    elif 'NN' in algorithm:
       k = int(algorithm.strip("NN"))
       for i in testing_input:
           results.append(KNN(k,training_input,i))

    for i in results:
        print(i)

