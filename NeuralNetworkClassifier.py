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
        # If there are hidden layers
        if self.num_hidden_layers > 0:
            # Runs either 1000 times or until it guesses everything correctly
            while not done and runs < 1000:
                # If this never changes, everything was predicted correctly
                done = True
                # Runs counter
                runs += 1
                # Loop through each row of data
                for index, data_row in enumerate(data):
                    # 2D array to keep track of nodes values at each layer
                    if self.num_hidden_layers > 1:
                        hidden_node_values = [[] for _ in range(self.num_hidden_layers - 1)]
                    else:
                        hidden_node_values = [[]]
                    # Set up the first layer with the data as inputs
                    for node in self.neural_network[0]:
                        hidden_node_values[0].append(node.train(data_row))
                    # Set up all the hidden ayers with the previous layer's activation as the value
                    for layer_index, layer in enumerate(self.neural_network[1:-2]):
                        for node in layer:
                            hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))
                    # A dictionary with the target name as the key and its activation as the value
                    target_values = dict()
                    # Gets the activation for each target
                    for node in self.neural_network[-1]:
                        target_values[node.target] = node.train(hidden_node_values[-1])
                    # If the highest activation value was the correct target, it predicted correctly!
                    if self.targets[index] != max(target_values.items(), key=operator.itemgetter(1))[0]:
                        # If did not guess correctly, we're going to have to loop again.
                        done = False
        # No hidden layers
        else:
            # Runs either 1000 times or if it guesses every target correctly
            while not done and runs < 1000:
                # If this never changes, everything was predicted correctly
                done = True
                # Runs counter
                runs += 1
                # Loop through each row of data
                for index, data_row in enumerate(data):
                    # A dictionary with the target name as the key and its activation as the value
                    target_values = dict()
                    # Loop through each node in the neural network and calculate its activation
                    for node in self.neural_network[0]:
                        target_values[node.target] = node.train(data_row)
                        # If the highest activation value was the correct target, it predicted correctly!
                        if self.targets[index] != max(target_values.items(), key=operator.itemgetter(1))[0]:
                            # If did not guess correctly, we're going to have to loop again.
                            done = False

    # Predicts the target for a particular row of data
    def predict(self, data_row):
        # If there are hidden layers
        if self.num_hidden_layers > 0:
            # 2D array to keep track of nodes values at each layer
            if self.num_hidden_layers > 1:
                hidden_node_values = [[] for _ in range(self.num_hidden_layers - 1)]
            else:
                hidden_node_values = [[]]
            # Set up the first layer with the data as inputs
            for node in self.neural_network[0]:
                hidden_node_values[0].append(node.train(data_row))
            # Set up all the hidden ayers with the previous layer's activation as the value
            for layer_index, layer in enumerate(self.neural_network[1:-2]):
                for node in layer:
                    hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))
            # A dictionary with the target name as the key and its activation as the value
            target_values = dict()
            # Gets the activation for each target
            for node in self.neural_network[-1]:
                target_values[node.target] = node.train(data_row)

            # Predicts the target with the highest activation value
            return max(target_values.items(), key=operator.itemgetter(1))[0]
        else: # No hidden layers
            # A dictionary with the target name as the key and its activation as the value
            target_values = dict()
            # Gets the activation for each target
            for node in self.neural_network[0]:
                target_values[node.target] = node.train(data_row)
            # Predicts the target with the highest activation value
            return max(target_values.items(), key=operator.itemgetter(1))[0]

class NeuralNetworkModel:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.model = []

    def predict(self, data):
        for data_row in data:
            self.model.append(self.neural_network.predict(data_row))

        return self.model
