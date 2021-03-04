"""AUTHOR: Patrick Callum O'Connell
Date: 11/10/2020 (date started)

This is the file that contains the actual class files for the neural net that the other two
driver classes use.

"""

import random
import math

class neuralNet:
    """A class that implements a perceptron-based neural network
    with no hidden layers. When instantiated, user can specify
    the starting weight, num of input nodes and num of output nodes"""

    def __init__(self, starting_weight, num_inputs, num_outputs):
        """initializes neural net with input and output arrays
        starting empty, and the weight array initialized with
        entered starting weight (an arbitrary float value)
        input number (int) and output number (int)
        """
        input_range = range(num_inputs)
        output_range = range(num_outputs)

        self.input = [0 for i in input_range]
        self.real_output = [0 for i in output_range]
        self.expected_output = [0 for i in output_range]
        self.weighted_input_sum = 0
        self.percent_correct = 0
        self.average_euclidean_distance = 0

        #creates 2d array for weights where first index is the input node
        #second index is the output node
        self.weights = [[starting_weight for j in output_range] for i in input_range]

    def initialize_inputs(self, input_list):
        """(list) --> none
        takes one training example list and initializes our input array with the values"""
        i = 0
        while (i < 64):
            self.input[i] = float(input_list[i])
            i += 1

    def initialize_expected_outputs(self, output_list):
        """(list) --> none
        takes one training example's expected output, initializes real outputs so that we can
        calculate error later"""
        i = 0
        while (i < 10):
            self.expected_output[i] = float(output_list[i])
            i += 1

    def init_weighted_input(self, output_num):
        """(int) --> none
        helper function that finds the sum of the weighted inputs for a given output (STARTS AT 0, NOT 1), stores it as value
        in this object"""

        weighted_sum = 0
        i = 0
        while (i < len(self.input)):
            weighted_sum += self.input[i] * self.weights[i][output_num]
            i += 1

        self.weighted_input_sum = weighted_sum

    def determine_outputs(self, learning_rate):
        """(int) --> none
        uses inputs (MUST already be initialized) to determine outputs
        also updates weights"""

        num_outputs = len(self.real_output)
        num_inputs = len(self.input)

        err_list = [0] * num_outputs
        g_prime_list = [0] * num_outputs

        #first, determines the outputs, storing the errors and g'(in) values in lists here
        j = 0
        while (j < num_outputs):
            self.init_weighted_input(j) #initalizes weighted input sum

            self.real_output[j] = 1 / (1 + math.exp((self.weighted_input_sum) * (-1)))
            err_list[j] = self.expected_output[j] - self.real_output[j]
            g_prime_list[j] = self.real_output[j] * (1.0 - self.real_output[j])

            j += 1
    
        #now, to update the weights
        i = 0
        while (i < num_inputs):

            #for each input, adjust the 10 associated weights
            j = 0
            while (j < num_outputs):
                self.weights[i][j] = self.weights[i][j] + (learning_rate * self.input[i] * err_list[j] * g_prime_list[j])
                j += 1
            
            i += 1
    
    def determine_outputs_wl(self):
        """(None) --> int
        uses inputs (MUST already be initialized) to determine outputs
        wl stands for weightless, meaning that this function does NOT
        update weights, for use in determining the error
        
        returns the euclidean distance for this specific example"""

        num_outputs = len(self.real_output)

        err_list = [0] * num_outputs

        #first, determines the outputs, storing the errors and g'(in) values in lists here
        j = 0
        while (j < num_outputs):
            self.init_weighted_input(j) #initalizes weighted input sum

            self.real_output[j] = 1 / (1 + math.exp((self.weighted_input_sum) * (-1)))
            err_list[j] = self.expected_output[j] - self.real_output[j]

            j += 1

        #finds euclidean distance from real output vector to expected output vector
        err_sum = 0
        for err in err_list:
            err_sum += (err)**2
        err_sum = math.sqrt(err_sum)

        return err_sum
    
    def determine_error(self, validation_list):
        """takes a validation list, runs it through this neural network
        and stores the average error and average euclidean distance in this object"""
        num_val_examples = len(validation_list)
        num_correct = 0
        euclid_error_sum = 0

        i = 0
        while (i < num_val_examples):
            
            self.initialize_inputs(validation_list[i][0])
            self.initialize_expected_outputs(validation_list[i][1])

            euclid_error = self.determine_outputs_wl()
            euclid_error_sum += euclid_error

            max_expected = max(self.expected_output)
            max_real = max(self.real_output)

            if (self.expected_output.index(max_expected) == self.real_output.index(max_real)):
                num_correct += 1

            i += 1
        
        
        self.percent_correct = num_correct / num_val_examples
        self.average_euclidean_distance = euclid_error_sum / num_val_examples


class exampleList:
    """A class that holds examples on which to train the neural net,
    also holds the validation set"""

    def __init__(self, example_text, percentage_test):
        """takes .txt file of examples, the desired percentage of those to be used
        in test set, and the desired percentage of those to be used in the validation set,
        goes through each example in the .txt file and either stores it in our list of test examples,
        or in the validation set, based on the given desired percentages as probabilities,
        but stops filling one set once we have reached our desired percentage"""

        desired_test_examples = percentage_test * 5620
        num_test_examples = 0

        percentage_validation = 1 - percentage_test
        desired_val_examples = percentage_validation * 5620
        num_val_examples = 0

        #these are lists of tuples, where the first part of the tuple is the inputs
        #and the second part is the correct output for that example
        self.train_list = []
        self.validation_list = []

        #loop goes through text file and extracts the examples, puts them into our example set
        #and validation set according to given percentages
        with open(example_text) as example:
            lines = example.readlines()

            i = 0 #iterator keeps track of current line, and helps index
            while (i < 5620*2):
                input_numbers = lines[i].split()[1:-1] #indexing at end chops off parentheses
                output_numbers = lines[i + 1].split()[1:-1]

                current_example = (input_numbers, output_numbers)

                #determines which set to add example to
                num = random.uniform(0, 1)
                if (num <= percentage_test) and (num_test_examples < desired_test_examples):
                    self.train_list.append(current_example)
                    num_test_examples += 1
                
                elif (num > percentage_test) and (num_val_examples < desired_val_examples):
                    self.validation_list.append(current_example)
                    num_val_examples += 1

                else:
                    self.train_list.append(current_example)
                    num_test_examples += 1

                i += 2 #moves iterator for next two lines (next example)


    
    

    

