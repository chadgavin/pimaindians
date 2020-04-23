import csv
import numpy as np 
import math
import sys

def extract(input_file):
    data = []
    lines = []
    with open(input_file,'r') as f:
        lines = f.readlines()
    lines = [x.strip() for x in lines]
    for i in lines:
        data.append(i.split(","))
    
    changing_to_float(data) 
    return data
def changing_to_float(data):
    for i in data:
        for j in range(len(i)):
            if j != 'yes' and j!= 'no':
                i[j] = np.float(i[j])



def main(argv):
	print('a')

if __name__ == "__main__":
   df =[]
   training_data = sys.argv[1]
   testing_data = sys.argv[2]
   algorithm = sys.argv[3]
   training_input = extract(training_data)
   #tesing_input = extract(testing_data)
   print(training_input)
   '''if algorithm == 'NB':
       NB(traing)'''