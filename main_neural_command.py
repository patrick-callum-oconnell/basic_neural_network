"""AUTHOR: Patrick Callum O'Connell
Date: 11/10/2020

This is the main driver that can be called from the command line.

Here are the parameters:
1 - split (percentage of examples used in training set (as opposed to the validation set))
2 - edge_weight (initial edge weights)
3 - epochs (how many epochs over which the neural network should train)
4 - example_file (the name of the .txt file that contains the examples)

"""

import sys
from train_net import train_neural_net

#creates and trains a neural net using given percentage of examples as training set
#given starting edge weight
#and given # of epochs

split = float(sys.argv[1])
edge_weight = float(sys.argv[2])
epochs = int(sys.argv[3])
example_file = str(sys.argv[4])

ourNeuralNetwork = train_neural_net(split, edge_weight, epochs, example_file)

print("For split " + str(split) + " as percentage of examples used in training set:")
print("Percentage correct: " + str(ourNeuralNetwork.percent_correct))
print("Average Euclidean distance: " + str(ourNeuralNetwork.average_euclidean_distance))
print("")


