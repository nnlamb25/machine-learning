class HardCodedClassifier:
    def __init__(self):
        pass

    def fit(self, data, target):
        return HardCodedModel()

class HardCodedModel:
    def __init__(self):
        self.model = []

    def predict(self, data):
        for row in data:
            self.model.append(0)

        return self.model