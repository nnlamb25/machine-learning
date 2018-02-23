import numpy as np
import random as rn
import operator
from math import exp

class NeuralNetworkCalssifier:
    def __init__(self):
        self.num_nodes = []
        self.num_hidden_layers = int(input("With how many hidden layers?\n> "))
        for i in range(self.num_hidden_layers):
            self.num_nodes.append(int(input("How many nodes in layer " + str(i + 1) + "?\n> ")))

    def fit(self, data, target):
        # We want the number of columns in the data set
        num_rows, num_cols = data.shape
        # Creates a neural network with num_cols nodes
        neural_network = Neurons(num_cols, self.num_hidden_layers, self.num_nodes, target)
        # Teaches the neural network from the data
        neural_network.teach(data)

        return NeuralNetworkModel(neural_network)


# A node which holds the weights between a neuron and the targets
class TargetVerticesNode:
    def __init__(self, num_inputs, target=None):
        # The weights between this neuron and every target
        self.input_weights = []
        # This is the target for this node
        self.target = target
        # The value this node carries (will be updated in train)
        self.value = 0
        # Accounts for a biased node
        self.bias = -1
        # Initially assigns random weights for each input and the biased node
        for _ in range(num_inputs + 1):
            self.input_weights.append(rn.uniform(-0.1, 0.1))

    #sigmoid function to determine whether or not the neuron fires
    def sigmoid(self, value):
        return 1 / (1 + exp(-value))

    # Trains this vertices node to have correct weights
    def train(self, data_row):#, data_target=None):
        self.value = 0
        n = -0.1
        # Gets the sum of the weights times the data input
        for index in range(len(data_row) - 1):
            self.value += data_row[index] * self.input_weights[index + 1]

        # Add the biased node
        self.value += self.bias * self.input_weights[0]
        self.value = self.sigmoid(self.value)

        return self.value

        #if data_target is None:
        #    return self.value
        #else:
        #    if (self.value >= 0.5 and data_target == self.target) or (self.value < 0.5 and data_target != self.target):
        #        return True
        #    else:
        #        return False


# Holds an array of vertices between the data inputs and their targets
class Neurons:
    def __init__(self, num_cols, num_hidden_layers, num_nodes, targets):
        # Gets the most common target
        unique, pos = np.unique(targets, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        self.most_common_target = targets[maxpos]
        # Will hold the vertices
        self.neural_network = [[] for _ in range(num_hidden_layers + 1)]
        # Holds all the targets for each data row
        self.targets = targets
        # Holds only the unique targets
        self.unique_targets = set(targets)
        # Number of hidden layers between input and output layers
        self.num_hidden_layers = num_hidden_layers
        # Create every layer including hidden layers
        if num_hidden_layers > 0:
            # Create first layer with the number of columns as the number of vertices
            for node in range(num_nodes[0]):
                self.neural_network[0].append(TargetVerticesNode(num_cols))

            # Create the hidden layers with the number of vertices being the number of nodes in the previous layer
            for index in range(num_hidden_layers - 1):
                for node in range(num_nodes[index + 1]):
                    self.neural_network[index + 1].append(TargetVerticesNode(num_nodes[index]))

            # Create he output layer
            for unique_target in self.unique_targets:
                self.neural_network[num_hidden_layers].append(TargetVerticesNode(num_nodes[-1], unique_target))

        else: # No hidden layers, only input and output
            for unique_target in self.unique_targets:
                self.neural_network[0].append(TargetVerticesNode(num_cols, unique_target))

    # Teaches the neuron array when to fire when given data
    def teach(self, data):
        # This will run when either all the weights are correct or after 1000 runs
        done = False
        runs = 0
        if self.num_hidden_layers > 0:
            while not done and runs < 1000:
                done = True
                runs += 1
                for index, data_row in enumerate(data):
                    hidden_node_values = [[] for _ in range(self.num_hidden_layers - 1)]
                    for node in self.neural_network[0]:
                        hidden_node_values[0].append(node.train(data_row))

                    for layer_index, layer in enumerate(self.neural_network[1:-2]):
                        for node in layer:
                            hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))

                    target_values = dict()
                    for node in self.neural_network[-1]:
                        target_values[node.target] = node.train(hidden_node_values[-1])

                    if self.targets[index] != max(target_values.items(), key=operator.itemgetter(1))[0]:
                        done = False
                    #for node in self.neural_network[-1]:
                    #    if not node.train(hidden_node_values[-1], self.targets[index]):
                    #        done = False

        else:
            while not done and runs < 1000:
                done = True
                runs += 1
                for index, data_row in enumerate(data):
                    target_values = dict()
                    for node in self.neural_array:
                        target_values[node.target] = node.train(data_row)

                        if self.targets[index] != max(target_values.items(), key=operator.itemgetter(1))[0]:
                            done = False
                        #if not node.train(data_row, self.targets[index]):
                        #    done = False

    # Predicts the target for a particular row of data
    def predict(self, data_row):
        if self.num_hidden_layers > 0:
            hidden_node_values = [[] for _ in range(self.num_hidden_layers - 1)]
            for node in self.neural_network[0]:
                hidden_node_values[0].append(node.train(data_row))

            for layer_index, layer in enumerate(self.neural_network[1:-2]):
                for node in layer:
                    hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))

            target_values = dict()
            for node in self.neural_network[-1]:
                target_values[node.target] = node.train(data_row)

            print(str(target_values) + " - " + str(max(target_values.items(), key=operator.itemgetter(1))[0]) + "\n")
            return max(target_values.items(), key=operator.itemgetter(1))[0]
        else:
            target_values = dict()
            for node in self.neural_network:
                target_values[node.target] = node.train(data_row)

            return max(target_values.items(), key=operator.itemgetter(1))[0]

class NeuralNetworkModel:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.model = []

    def predict(self, data):
        for data_row in data:
            self.model.append(self.neural_network.predict(data_row))

        return self.model
