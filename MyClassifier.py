#Source referenced from:
#  https://towardsdatascience.com/na%C3%AFve-bayes-from-scratch-using-python-only-no-fancy-frameworks-a1904b37222d
#  https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
#  https://dzone.com/articles/naive-bayes-tutorial-naive-bayes-classifier-in-pyt
#  https://www.edureka.co/blog/naive-bayes-tutorial/
import csv
import numpy as np 
from math import sqrt
from math import pi
from math import exp
import math 
import sys
import random
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

def pdf(x,mean,sd):
    try:
        value_of_exp = (((x-mean)/sd) **2) * (-1/2)
        value = (1/(sd * sqrt(2 * pi)))* exp(value_of_exp)
        return value
    except ZeroDivisionError:
        return 0

def class_prob(summaries,row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= pdf(row[i], mean, stdev)
    return probabilities

def split(filename):
    a = extract(filename)
    yes_list = []
    no_list = []
    count = len(a)
    count_of_yes = 0
    count_of_no = 0
    folds = 10
    for i in a:
        if i[-1] == 1:
            count_of_yes +=1
            yes_list.append(i)
        elif i[-1] == 0:
            no_list.append(i)
            count_of_no +=1
    
    no_of_entries_in_each_fold = count/folds
    no_of_yes_in_each_fold = math.floor(count_of_yes/folds)
    no_of_no_in_each_fold = math.floor(count_of_no/folds)

    random.shuffle(yes_list)
    random.shuffle(no_list)

    buckets = [[] for i in range(10)]
    for i in range(10):
        for num_yes in range(no_of_yes_in_each_fold):
            holder = yes_list.pop()
            holder[-1] = "yes"
            buckets[i].append(holder)
        for num_no in range(no_of_no_in_each_fold):
            holder = no_list.pop()
            holder[-1] = "no"
            buckets[i].append(holder)
        random.shuffle(buckets[i])

# Ensure the remainder of the elements
    if yes_list:
        for i in range(len(yes_list)):
            holder = yes_list.pop()
            holder[-1] = "yes"
            buckets[i].append(holder)
            random.shuffle(buckets[i])
    elif no_list:
        for i in range(len(no_list)):
            holder = no_list.pop()
            holder[-1] = "no"
            buckets[i].append(holder)
            random.shuffle(buckets[i])

    try:
        file = open("pima-folds.csv",mode= 'w',newline='')

        with file:
            write = csv.writer(file)
            for i in range(len(buckets)):
                if i>0:
                    write.writerow('fold' + str(i+1))
                else:
                    write.writerow('fold' + str(i + 1))
                for j in buckets[i]:
                    write.writerow(j)
    except:
        print("Error with Pima-folds.csv")


def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
    sd = sqrt(variance)
    return sd

def predict(summaries, row):
    probabilities = class_prob(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
        if probability == 0.5:
            return 1
    
    return best_label


def NB(training_input,tesing_input):
    seperated = summarize_by_class(training_input)
    result = []
    for i in tesing_input:
        outcome = predict(seperated,i)
        if outcome == 1:
            print('yes')
        elif outcome == 0:
            print('no')
       
      

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

def average_cal(itr):
    return sum(itr) / len(itr)

def strip_end(data):
    result = []
    for i in range(len(data)):
        result.append(data[i][-1])
        data[i] = data[i][:-1]

    return data,result


def ten_fold_cross_validation(filename,algo):

    lines = extract(filename)

    input_buckets = [[] for i in range(10)]
    checker = False
    k = 0
    for line in lines[1:]:
        if line.count(0) == 9 and checker == False:
            k+=1
            checker = True
            continue
        elif line.count(0) == 9 and checker == True:
            checker = False
            continue
        else:
            input_buckets[k].append(line)

    overall_accuracy = []

    for i in range(10):
        internal_results = []
        testing_set,actual_result = strip_end(input_buckets[i])
        if i != 9:
            training_set = input_buckets[:i] + input_buckets[i+1:]
        else:
            training_set = input_buckets[:i]

        #flatten training_set so that its a row of values only
        training_set = [e for sl in training_set for e in sl]

        if algo == 'NB':
            internal_results.append(NB(training_set,testing_set))

        elif 'NN' in algo:
            k = int(algorithm.strip("NN"))
            for input in testing_set:
                internal_results.append(KNN(k, training_set, input))

        sum_correct = 0
        num_examples = len(internal_results)
        for i in range(len(internal_results)):
            if internal_results[i] == 'yes' and actual_result[i] == 1:
                sum_correct +=1
            elif internal_results[i] == 'no' and actual_result[i] == 0:
                sum_correct += 1
            else:
                continue

        accuracy = sum_correct/num_examples
        overall_accuracy.append(accuracy)
        sum_correct = 0

    return average_cal(overall_accuracy)


def main(argv):
	print('a')

if __name__ == "__main__":
    results =[]
    training_data = sys.argv[3]
    testing_data = sys.argv[4]
    algorithm = sys.argv[5]
    training_input = extract(training_data)
    testing_input = extract(testing_data)
    # split("pima.csv")

    print(ten_fold_cross_validation(training_data,algorithm))

    if algorithm == 'NB':
       result = NB(training_input,testing_input)
    elif 'NN' in algorithm:
       k = int(algorithm.strip("NN"))
       for i in testing_input:
           results.append(KNN(k,training_input,i))

    for i in results:
        print(i)
