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
        # The error of this node (will be updated)
        self.delta = 0
        # Accounts for a biased node
        self.bias = -1
        # Initially assigns random weights for each input and the biased node
        for _ in range(num_inputs + 1):
            self.input_weights.append(rn.uniform(-0.1, 0.1))

    # sigmoid function to determine whether or not the neuron fires
    @staticmethod
    def sigmoid(value):
        return 1 / (1 + exp(-value))

    # Trains this vertices node to have correct weights
    def train(self, data_row):
        self.value = 0
        # Gets the sum of the weights times the data input
        for index in range(len(data_row)):
            self.value += data_row[index] * self.input_weights[index + 1]

        # Add the biased node
        self.value += self.bias * self.input_weights[0]
        self.value = self.sigmoid(self.value)

        return self.value


# Holds an array of vertices between the data inputs and their targets
class Neurons:
    def __init__(self, num_cols, num_hidden_layers, num_nodes, targets):
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
                for _ in range(num_nodes[index + 1]):
                    self.neural_network[index + 1].append(TargetVerticesNode(len(self.neural_network[index])))

            # Create he output layer
            for unique_target in self.unique_targets:
                self.neural_network[num_hidden_layers].append(TargetVerticesNode(
                    len(self.neural_network[num_hidden_layers - 1]), unique_target))

        else:  # No hidden layers, only input and output
            for unique_target in self.unique_targets:
                self.neural_network[0].append(TargetVerticesNode(num_cols, unique_target))

    # Teaches the neuron array when to fire when given data
    def teach(self, data):
        # This will run when either all the weights are correct or after 1000 runs
        done = False
        runs = 0
        accuracy = self.get_accuracy(data)
        print("Starting accuracy: " + str(round(accuracy * 100, 3)) + "%\n")
        # If there are hidden layers
        if self.num_hidden_layers > 0:
            # Runs either 1000 times or until it guesses everything correctly
            while not done and runs < 10000:
                # If this never changes, everything was predicted correctly
                done = True
                # Runs counter
                runs += 1
                # Loop through each row of data
                for index, data_row in enumerate(data):
                    # 2D array to keep track of nodes values at each layer
                    if self.num_hidden_layers > 1:
                        hidden_node_values = [[] for _ in range(self.num_hidden_layers)]
                    else:
                        hidden_node_values = [[]]
                    # Set up the first layer with the data as inputs
                    for node in self.neural_network[0]:
                        hidden_node_values[0].append(node.train(data_row))
                    # Set up all the hidden ayers with the previous layer's activation as the value
                    for layer_index, layer in enumerate(self.neural_network[1:-1]):
                        for node in layer:
                            hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))
                    # A dictionary with the target name as the key and its activation as the value
                    target_values = dict()
                    # Gets the activation for each target
                    for node in self.neural_network[-1]:
                        target_values[node.target] = node.train(hidden_node_values[-1])
                    # Get the target with the highest activation value
                    prediction = self.get_key_with_max_value(target_values)
                    # If the highest activation value was the correct target, it predicted correctly!
                    if self.targets[index] != prediction:
                        # If we didn't predict correctly, we need to recalculate these weights
                        self.recalculate_node_values(prediction, self.targets[index], data_row)
                        # We're going to have to loop again.
                        done = False
                if runs % 2000 == 0:
                    accuracy = self.get_accuracy(data)
                    print("Now " + str(round(accuracy * 100, 3)) + "% accurate. - " + str(runs) + "\n")
        else:  # No hidden layers
            # Runs either 1000 times or if it guesses every target correctly
            while not done and runs < 10000:
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
                    # Get the target with the highest activation value
                    prediction = self.get_key_with_max_value(target_values)
                    # If the highest activation value was the correct target, it predicted correctly!
                    if self.targets[index] != prediction:
                        # If we didn't predict correctly, we need to recalculate these weights
                        self.recalculate_node_values(prediction, self.targets[index], data_row)
                        # If did not guess correctly, we're going to have to loop again.
                        done = False
                if runs % 2000 == 0:
                    accuracy = self.get_accuracy(data)
                    print("Now " + str(round(accuracy * 100, 3)) + "% accurate. - " + str(runs) + "\n")

    # Recalculates the weights
    def recalculate_node_values(self, wrongly_predicted_target, correct_target, data):
        # Recalculates the deltas for the required nodes
        self.recalculate_deltas(wrongly_predicted_target, correct_target)
        # If there are hidden layers
        if self.num_hidden_layers > 0:
            # Loop through the target nodes and change the weights for the required targets
            for node_index, node in enumerate(self.neural_network[-1]):
                # Only need to change the weights of the targets that should have been
                # predicted and weren't or that were wrongly predicted
                if node.target == correct_target or node.target == wrongly_predicted_target:
                    # Reassign the weights of the node
                    self.neural_network[-1][node_index] = self.calc_weights(node, self.neural_network, -2)
            # Now loop through every layer between the output layer and the first hidden layer
            for layer_index, layer in enumerate(self.neural_network[1:-1]):
                for node_index, node in enumerate(layer):
                    # Recalculate the node's weights
                    self.neural_network[layer_index + 1][node_index] = self.calc_weights(node, self.neural_network,
                                                                                     layer_index)
        # Recalculate the weights for the first hidden layer (or only layer if no hidden layers)
        # The data is the input this time, no previous nodes to get values from
        for node_index, node in enumerate(self.neural_network[0]):
            self.neural_network[0][node_index] = self.calc_weights(node, data)

    # Recalculate the weights of a particular node.  layer_index will determine which layer of the neural
    # network this node resides, unless it is on the first (or only) layer, in which case there will be no layer_index
    @staticmethod
    def calc_weights(node, values, prev_layer_index=None):
        # n used for calculating new weight
        n = -0.1
        # This is the first (or only) layer in the neural network.
        if prev_layer_index is None:
            # Loop through all the weights of vertices that are attached to input values
            for weight_index in range(len(node.input_weights) - 1):
                # Reassign the node's weight
                node.input_weights[weight_index + 1] = node.input_weights[weight_index + 1] - (
                        n * node.delta * values[weight_index])
            # Calculate the new weight for the bias node
            node.input_weights[0] = node.input_weights[0] - (n * node.delta * node.bias)
        else:  # This node resides in a hidden layer that isn't the first layer
            # Loop through all the weights of vertices that are attached to input values
            for weight_index in range(len(node.input_weights) - 1):
                # Reassign the node's weight
                node.input_weights[weight_index + 1] = node.input_weights[weight_index + 1] - (
                        n * node.delta * values[prev_layer_index][weight_index].value)
            # Calculate the new weight for the bias node
            node.input_weights[0] = node.input_weights[0] - (n * node.delta * node.bias)
        # Return the node with the new updated weights
        return node

    # Recalculates the required deltas for the nodes
    def recalculate_deltas(self, wrongly_predicted_target, correct_target):
        # Loop through each target node and recalculate its delta if it needs to be recalculated
        for target_node_index, target_node in enumerate(self.neural_network[-1]):
            # This is what it was supposed to predict and didn't
            if target_node.target == correct_target:
                # Calculate the error of the target node
                target_node.delta = target_node.value * (1 - target_node.value) * (target_node.value - 1)
            # All other nodes
            elif target_node.target == wrongly_predicted_target:  # target_node is not the correct target value
                # Calculate the error
                target_node.delta = target_node.value * (1 - target_node.value) * target_node.value
            # Assign the new node with the updated delta to the neural network
            self.neural_network[-1][target_node_index] = target_node
        # If there are more hidden layers, we're not done yet
        if self.num_hidden_layers > 0:
            # We need to start at the hidden layers closest to the target and work backwards
            hidden_layer_index = len(self.neural_network) - 2
            # Go through all the hidden layers from here
            while hidden_layer_index >= 0:
                # Loop through each node in this hidden layer
                for node_index, node in enumerate(self.neural_network[hidden_layer_index]):
                    # Calculate the sum for each node next layer node's delta times the weight to that node
                    sum_delta_weights = 0
                    for prev_node in self.neural_network[hidden_layer_index + 1]:
                        sum_delta_weights += prev_node.input_weights[node_index + 1] * prev_node.delta
                    # Calculate the new delta for the node
                    node.delta = node.value * (1 - node.value) * sum_delta_weights
                    # Assign the new node with the updated delta to the neural network
                    self.neural_network[hidden_layer_index][node_index] = node
                # Decrement the index
                hidden_layer_index -= 1

    # Gets the key of the item with the max value in a dictionary
    @staticmethod
    def get_key_with_max_value(dictionary):
        return max(dictionary.items(), key=operator.itemgetter(1))[0]

    # Gets the accuracy of the current iteration
    def get_accuracy(self, data):
        num_predicted_correctly = 0
        for index, data_row in enumerate(data):
            # print(str(self.predict(data_row)) + " - " + str(self.targets[index]))
            if self.predict(data_row) == self.targets[index]:
                num_predicted_correctly += 1

        return num_predicted_correctly / len(self.targets)

    # Predicts the target for a particular row of data
    def predict(self, data_row):
        # If there are hidden layers
        if self.num_hidden_layers > 0:
            # 2D array to keep track of nodes values at each layer
            if self.num_hidden_layers > 1:
                hidden_node_values = [[] for _ in range(self.num_hidden_layers)]
            else:
                hidden_node_values = [[]]
            # Set up the first layer with the data as inputs
            for node in self.neural_network[0]:
                hidden_node_values[0].append(node.train(data_row))
            # Set up all the hidden ayers with the previous layer's activation as the value
            for layer_index, layer in enumerate(self.neural_network[1:-1]):
                for node in layer:
                    hidden_node_values[layer_index + 1].append(node.train(hidden_node_values[layer_index]))
            # A dictionary with the target name as the key and its activation as the value
            target_values = dict()
            # Gets the activation for each target
            for node in self.neural_network[-1]:
                target_values[node.target] = node.train(hidden_node_values[-1])
            # Predicts the target with the highest activation value
            return self.get_key_with_max_value(target_values)
        else:  # No hidden layers
            # A dictionary with the target name as the key and its activation as the value
            target_values = dict()
            # Gets the activation for each target
            for node in self.neural_network[0]:
                target_values[node.target] = node.train(data_row)
            # Predicts the target with the highest activation value
            return self.get_key_with_max_value(target_values)


class NeuralNetworkModel:
    def __init__(self, neural_network):
        self.neural_network = neural_network
        self.model = []

    def predict(self, data):
        for data_row in data:
            self.model.append(self.neural_network.predict(data_row))

        return self.model
