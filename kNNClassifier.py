import numpy
import math

class kNNClassifier:
    def __init__(self, k):
        self.k = k
        pass

    def fit(self, data, target):
        return kNNModel(data, target, self.k)


class kNNModel:
    def __init__(self, data, target, k):
        self.data = data
        self.target = target
        self.k = k

    def predict(self, test_data):
        return self.kNearestNeighbors(test_data)

    def kNearestNeighbors(self, test_data):
        nInputs = numpy.shape(test_data)[0]
        closest = numpy.zeros(nInputs)
        for n in range(nInputs):
            # Calculate Distance
            distances = numpy.sum((self.data - test_data[n, :]) ** 2, axis=1)
            # Get nearest neighbor
            indices = numpy.argsort(distances, axis=0)

            classes = numpy.unique(self.target[indices[:self.k]])
            if len(classes) == 1:
                closest[n] = numpy.unique(classes)
            else:
                counts = numpy.zeros(max(classes) + 1)
                for i in range(self.k):
                    counts[self.target[indices[i]]] += 1
                closest[n] = numpy.max(counts)

        return closest
