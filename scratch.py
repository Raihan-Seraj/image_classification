# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 18:49:21 2017

@author: Hp
"""

from random import seed
from random import random
from math import exp, tanh
import numpy as np
from itertools import repeat

x = np.loadtxt("train_x.csv", delimiter=",") # load from text 
y = np.loadtxt("train_y.csv", delimiter=",") 
#x = x.reshape(-1, 64, 64) # reshape 

print("Data loaded")

dataset = [[] for i in range(50000)]
for i in range(50000):
    for j in range(len(x[i])):
        dataset[i].append(x[i][j])
O = []
ljk = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 24, 25, 27, 28, 30, 32, 35, 36, 40, 42, 45, 48, 49, 54, 56, 63, 64, 72, 81]
for i in range(len(ljk)):
    O.append(ljk.index(ljk[i]))     
    
y = y.reshape(-1, 1) 
y = y[0:50]

for i in range(len(y)):
    temp = y[i][0]
    dataset[i].append(temp)
 
print("Dataset created")
 
# Rescale dataset columns to the range 0-1
def dataset_minmax(dataset):
    minmax = list()
    minmax = [[min(column), max(column)] for column in zip(*dataset)]
    del(minmax[-1])
    normalized = []
    for row in dataset:       
        for i in range(len(row)-1):
            row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
            normalized.append(row[i])
    chunks = [normalized[x:x+2] for x in range(0, len(normalized), 2)]
    return chunks


def initialize_network(n_inputs, n_hidden1, n_hidden2, n_outputs):
    network = list()
    hidden_layer1 = [{'weights' :[random() for i in range(n_inputs+1)]} for i in range(n_hidden1)]
    network.append(hidden_layer1)
    hidden_layer2 = [{'weights' :[random() for i in range(n_hidden1+1)]} for i in range(n_hidden2)]
    network.append(hidden_layer2)
    output_layer = [{'weights':[random() for i in range(n_hidden2+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network



def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights)-1):
        activation += weights[i]*inputs[i]
    return activation


def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, row):
    inputs =row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs



def transfer_derivative(output):
    return output * (1.0 - output)


def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network)-1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i+1]:
                    error += (neuron['weights'][j]*neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j]*transfer_derivative(neuron['output'])



def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i!= 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate*neuron['delta']*inputs[j]
            neuron['weights'][1] += l_rate*neuron['delta']
            
 
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error=0
        for row in (train):
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]            
            expected[row[-1]] = 1
            print(expected)   
            sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network,expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


def predict(network, row):
    outputs = forward_propagate(network,row)    
    return outputs.index(max(outputs))


     
#Calling the NN    
n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))
n_hidden1 = 500
n_hidden2 = 500

normalized_input = dataset_minmax(dataset)

for i in range(len(normalized_input)):
    temp = normalized_input[i]
    temp.append(dataset[i][-1])

dataset = normalized_input

train_set = dataset[0:35000]

network = initialize_network(n_inputs, n_hidden1, n_hidden2, n_outputs)
train_network(network, train_set, 0.1, 20000, n_outputs)

print("Training done")

for i in range(len(network)):
    for j in range(len(network[i])):
        d = network[i][j]
        try:
            del d['delta']
            del d['output']
        except KeyError:
            pass

    
test_set = dataset[35000:50000]
minmax = list()
minmax = [[min(column), max(column)] for column in zip(*dataset)]
del(minmax[-1])    
normalized = []
for row in test_set:       
    for i in range(len(row)-1):
        row[i]=(row[i]-minmax[i][0])/(minmax[i][1]-minmax[i][0])
        normalized.append(row[i])
chunks = [normalized[x:x+4096] for x in range(0, len(normalized), 4096)]      



for row in test_set:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], O[prediction]))
    
	 
