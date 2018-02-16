import numpy as np
import random as rn

class NeuralNetworkCalssifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        # We want the number of columns in the data set
        num_rows, num_cols = data.shape
        # Creates a neural network with num_cols nodes
        neural_network = Neurons(num_cols, target)
        # Teaches the neural network from the data
        neural_network.teach(data)

        return NeuralNetworkModel(target)


class NeuralNetworkModel:
    def __init__(self, target):
        self.target = target
        self.model = []

    def predict(self, data):
        unique, pos = np.unique(self.target, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        most_common_target = self.target[maxpos]
        for _ in data:
            self.model.append(most_common_target)

        return self.model


# A node which holds the weights between a neuron and the targets
class TargetVerticesNode:
    def __init__(self, num_inputs, threshold, target):
        # The weights between this neuron and every target
        self.input_weights = []
        # This is the value the neuron uses to compare its value to determine if it should fire
        self.threshold = threshold
        # This is the target for this node
        self.target = target
        # Initially assigns random weights
        for _ in range(num_inputs):
            self.input_weights.append(rn.uniform(-0.1, 0.1))

    # Trains this vertices node to have correct weights
    def train(self, data_row, data_target):
        value = 0
        n = -0.1
        #print(self.input_weights)
        # Gets the sum of the weights times the data input
        for index in range(len(data_row)):
            value += data_row[index] * self.input_weights[index]

        print("VALUE FOR " + str(data_row) + " : " + str(value))
        return True
"""
        # Node fired (predicting this data row is for this node's target
        if value >= self.threshold and value < self.threshold + 2:
            # Fired correctly
            if data_target == self.target:
                #print("TRUE - FIRED CORRECTLY")
                return True
            # Fired and it shouldn't have
            else:
                #print("FALSE - FIRED AND IT SHOULDN'T HAVE")
                index = 0
                for weight in self.input_weights:
                    self.input_weights[index] = weight - n * (value - self.threshold) * data_row[index]
                    index += 1
                return False
        # Node did not fire (predicting this data row is not for this node's target
        else:
            # Did not fire correctly
            if data_target != self.target:
                #print("TRUE - DID NOT FIRE CORRECTLY")
                return True
            # Did not fire and it should have
            else:
                #print("FALSE - DID NOT FIRE AND IT SHOULD HAVE")
                index = 0
                for weight in self.input_weights:
                    self.input_weights[index] = weight - n * (self.threshold - value) * data_row[index]
                    index += 1
                return False
"""

# Holds an array of vertices between the data inputs and their targets
class Neurons:
    def __init__(self, num_cols, targets):
        # Will hold the vertices
        self.neural_array = []
        # Holds all the targets for each data row
        self.targets = targets
        # Holds only the unique targets
        self.unique_targets = set(targets)
        # Determines the threshold based on the number of targets, each Separated by a distance of 2
        thresholds = np.arange(-len(self.unique_targets), len(self.unique_targets), 2)
        # Creates a vertices node for each unique target
        index = 0
        for unique_target in self.unique_targets:
            self.neural_array.append(TargetVerticesNode(num_cols, thresholds[index], unique_target))
            index += 1

    # Teaches the neuron array when to fire when given data
    def teach(self, data):
        # This will run when either all the weights are correct or after 1000 runs
        done = False
        runs = 0
        while not done and runs < 1000:
           # if runs % 100 == 0:
            #    print(runs)
            done = True
            runs += 1
            for data_row in data:
                index = 0
                for node in self.neural_array:
                    if not node.train(data_row, self.targets[0]):
                        done = False

                index += 1

       # print("\n\nDONE")
       # print("CORRECT: " + str(done) + " - RUNS: " + str(runs))
