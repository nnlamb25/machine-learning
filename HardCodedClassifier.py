import numpy as np

class HardCodedClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        return HardCodedModel(target)


class HardCodedModel:
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
