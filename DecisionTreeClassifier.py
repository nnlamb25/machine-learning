import numpy as np
import math

class DecisionTreeClassifier:
    def __init__(self):
        pass

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def fit(self, data, target):
        self.target = target
        tree = self.make_tree(data, target, self.feature_names)
        print(tree)

        return DecisionTreeModel(data, target)

    # Returns the most freqquent target
    def most_frequent_target(self):
        unique, pos = np.unique(self.target, return_inverse=True)
        counts = np.bincount(pos)
        maxpos = counts.argmax()
        return self.target[maxpos]

    def calc_entropy(self, p):
        if p != 0:
            return -p * math.log2(p)
        else:
            return 0

    def get_feature_values(self, data, feature):
        nData = len(data)
        # List the values that feature can take
        values = []
        for datapoint in data:
            if datapoint[feature] not in values:
                values.append(datapoint[feature])

        return values

    def calc_info_gain(self, data, target, feature):
        gain = 0
        nData = len(data)
        values = self.get_feature_values(data, feature)
        featureCounts = np.zeros(len(values))
        entropy = np.zeros(len(values))
        valueIndex = 0
        # Find where those values appear in data[feature] and the corresponding target
        for value in values:
            dataIndex = 0
            newClasses = []
            for datapoint in data:
                if datapoint[feature] == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(target[dataIndex])
                dataIndex += 1

            # Get the values in new targets
            classValues = []
            for aclass in newClasses:
                if classValues.count(aclass) == 0:
                    classValues.append(aclass)

            classCounts = np.zeros(len(classValues))
            classIndex = 0
            for classValue in classValues:
                for aclass in newClasses:
                    if aclass == classValue:
                        classCounts[classIndex] += 1

                classIndex += 1

            for classIndex in range(len(classValues)):
                entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex]) / sum(classCounts))

            gain += float(featureCounts[valueIndex] / nData * entropy[valueIndex])
            valueIndex += 1

        #print("GAIN IN")
        #print(gain)
        return gain

    def make_tree(self, data, target, featureNames):
        # Various initialisations suppressed
        newData = np.empty(data)
        newTarget = np.empty(target)
        newNames = np.empty(featureNames)
        nData = len(data)
        nFeatures = len(featureNames)

        # If there is no more data and no more features, return the most frequent value
        print("DATA LEFT: " + str(nData) + " FEATURES LEFT: " + str(nFeatures))
        if nData == 0 and nFeatures == 0:
            if len(target) != 0:
                target_set = set(target)
                frequency = [0] * len(target_set)
                index = 0
                for value in target_set:
                    frequency[index] = np.count_nonzero(target == value)
                    # frequency[index] = target.count(value)
                    index += 1

                default = target[np.argmax(frequency)]
            else:
                default = self.most_frequent_target()

            return default
        elif np.count_nonzero(target[0] == nData): #target.count(target[0]) == nData:
            return target[0]
        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            values = []
            for feature in range(nFeatures):
                gain[feature] = self.calc_info_gain(data, target, feature)
                # Find possible feature values
                values.extend(self.get_feature_values(data, feature))

            bestFeature = np.argmin(gain)
            tree = {featureNames[bestFeature]: {}}

            # Find where those values appear in data[feature] and the corresponding class
            for value in values:
                index = 0
                # Find the datapoints with each feature value
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            datapoint = datapoint[1:]
                            newNames = featureNames[1:]
                            print("INSIDE FIRST\nDATAPOINT")
                            print(datapoint)
                            print("NEW NAMES")
                            print(newNames)
                        elif bestFeature == nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                            print("INSIDE SECOND\nDATAPOINT")
                            print(datapoint)
                            print("NEW NAMES")
                            print(newNames)
                        else:
                            datapoint = datapoint[:bestFeature]
                            np.append(datapoint, datapoint[bestFeature + 1:])#datapoint.extend(datapoint[bestFeature + 1:])
                            newNames = featureNames[:bestFeature]
                            np.append(newNames, featureNames[bestFeature + 1:])#newNames.extend(featureNames[bestFeature + 1:])
                            print("INSIDE THIS ONE!!\nDATAPOINT")
                            print(datapoint)
                            print("NEW NAMES")
                            print(newNames)

                        print("DATAAAAAAA")
                        print(datapoint)
                        np.append(newData, datapoint)#newData.append(datapoint)
                        np.append(newTarget, target[index])#newTarget.append(target[index])
                        print("\nTHIS IS THE NEW DATA")
                        print(newData)
                        print("\nTHIS IS THE NEW TARGET")
                        print(newTarget)
                    index += 1
                # Now recurse to the next level
                print("\n\nNEW DATA")
                print(newData)
                print("NEW TARGET")
                print(newTarget)
                print("NEW NAMES")
                print(newNames)
                subtree = self.make_tree(newData, newTarget, newNames)
                # And on returning, add the subtree on to the tree
                tree[featureNames[bestFeature]][value] = subtree
            return tree


class DecisionTreeModel:
    def __init__(self, data, target):
        self.data = data
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
