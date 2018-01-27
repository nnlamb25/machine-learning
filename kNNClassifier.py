import numpy

# Classifier for Nearest Neighbor Algorithm
class kNNClassifier:
    def __init__(self, k):
        self.k = k
        pass

    # Fits the data for the model
    def fit(self, data, target):
        return kNNModel(data, target, self.k)

# Model for the Nearest Neighbor Algorithm
class kNNModel:
    def __init__(self, data, target, k):
        self.data = data
        self.target = target
        self.k = k

    # returns the predicted values from the given test and target values
    def predict(self, test_data):
        return self.kNearestNeighbors(test_data)

    # returns an array of the nearest neighbors
    def kNearestNeighbors(self, test_data):
        # Gets the number of tests
        nInputs = numpy.shape(test_data)[0]

        # Creates an empty array of length nInputs, this will hold the closest values
        closest = numpy.empty(nInputs, dtype=object)

        # Used as a counter to let the user know the program is still running
        num_computes = 0

        for n in range(nInputs):
            # Calculate Distance
            distances = numpy.sum((self.data - test_data[n, :]) ** 2, axis=1)

            # Every thousand computations the message "Computing..." will display
            # so the user knows the program is still running.
            num_computes += 1
            if num_computes % 1000 == 0:
                print("Computing...")

            # Gets the indices of the sorted list of distances
            indices = numpy.argsort(distances, axis=0)

            # Get nearest neighbors within k distance, only looking at unique distances
            classes = numpy.unique(self.target[indices[:self.k]])

            # If there is only one neighbor closest, joins that group
            if len(classes) == 1:
                closest[n] = classes[0]

            # Otherwise, figure out which of the closest neighbors
            # appears most often and joins that group
            else:
                # Creates a dictionary with the key being the closest targets
                counts = dict()
                for class_value in classes:
                    counts[class_value] = 0

                # Gives the closest targets a count based on how many are k close
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1

                # The closest target was found!  Append it to the closest array
                closest[n] = max(counts, key=counts.get)

        # Returns the closest predicted values
        return closest
