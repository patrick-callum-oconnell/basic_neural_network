# basic_neural_network

AUTHOR: Patrick O'Connell
Date: 11/10/2020

This is a perceptron-based neural network model built from scratch in python. By "from scratch," I mean that no neural network related libraries were used. This neural network is written only with the help of the "random" and "math" libraries.

The structure of this model is a two-layer neural network that takes in a vector representation of an 8x8 picture of a black and white hand drawn number, and outputs what number the model thinks it is. So, the first layer is made of 64 neurons which each take a value from 0.0-1.0, where 0.0 is completely white, and 1.0 is completely black. The output layer is made of 10 neurons, each representing an integer from 0-9 (which is what the network thinks the number is).

Here is an example of a single example the program would use to train:
( 0 0 0.45 0.9 0.5625 0.0625 0 0 0 0 0.8125 0.9375 0.625 0.9375 0.3125 0 0 0.1956 0.9875 0.125 0 0.6875 0.5 0 0 0.25 0.75 0 0 0.5 0.5 0 0 0.3345 0.5 0 0 0.5565 0.5 0 0 0.25 0.6875 0 0.0625 0.75 0.4375 0 0 0.125 0.875 0.3125 0.625 0.75 0 0 0 0 0.375 0.8125 0.625 0 0 0 )
( 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 )

For the network to train properly, the examples MUST be in this format.

The driver (main_neural_command.py) takes in the desired values for the neural network training as well as the example set, trains the network, and then outputs the accuracy and other resultant data once the network has been sufficiently trained.

DESCRIPTION OF FILES:

classes.py

classes.py contains the neuralNet and exampleList classes. The neuralNet class defines a neuralNet object which holds values for input nodes, output nodes, and edge weights (as well as some other
values that are used in various methods that update weights, determine the error, etc). The exampleList class defines and object that provides functionality for extracting examples from a test file given a split percentage and
holds the lists of examples for the training set and validation set.

train_net.py

train_net.py defines the function train_neural_net that takes our training set split percentage, initial edge weight, and number of epochs, and uses those parameters to train and return a neural network.

main_neural_command.py

main_neural_command.py will train a neural network given the split percentage (% wanted for training set), edge_weight, number of epochs, and text file name as COMMAND LINE / TERMINAL arguments in that order.

HOW TO RUN:

Use the command "python main_neural_command.py arg1 arg2 arg3 arg4" in COMMAND PROMPT or TERMINAL
where arg1 = split %, arg2 = initial edge weight, arg3 = number of epochs, and arg4 is the name of the text file that holds the examples.
Results will be printed out.
