#AUTHOR: Patrick O'Connell
#Date: 11/10/2020

from classes import neuralNet, exampleList

def train_neural_net(training_percentage, starting_edge_weight, epochs, example_file):
    """instantiates and trains a neuralNet object given the percentage of examples that will be in the training set,
    and an initial edge weight, returns that neuralNet object"""

    #splits set into 70% training, 30% validation
    all_examples = exampleList(example_file, training_percentage)
    num_examples = len(all_examples.train_list)

    #initializes the neural net with starting weight .1, 64 input nodes
    #10 output nodes
    ourNeuralNet = neuralNet(starting_edge_weight, 64, 10)

    print("Beginning training...")

    #trains the neural network through number of specified epochs
    i = 0
    while (i < epochs):

        #print("Epoch: " + str(i))

        j = 0 #useful for indexing examples, not just an iterator
        while (j < num_examples):

            #initializes the inputs and expected output in our actual neural net
            ourNeuralNet.initialize_inputs(all_examples.train_list[j][0])
            ourNeuralNet.initialize_expected_outputs(all_examples.train_list[j][1])

            #determines outputs and updates weights w/ learning rate 0.01
            #note: determine outputs also adjusts weights
            ourNeuralNet.determine_outputs(0.01)

            j += 1
        
        i += 1
    
    print("Training completed successfully.")

    ourNeuralNet.determine_error(all_examples.validation_list)

    return ourNeuralNet




