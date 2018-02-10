import numpy as np
import math

class DecisionTreeClassifier:
    def __init__(self):
        self.num_passes = 0
        pass

    def set_feature_names(self, feature_names):
        self.feature_names = feature_names

    def fit(self, data, target):
        self.target = target
        tree = self.make_tree(data, target, self.feature_names)

        return DecisionTreeModel(tree, self.feature_names)

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
        # List the values that feature can take
        values = []
        if len(data) == 1 and data not in values:
            values.append(data)
        elif len(data) > 1:
            for datapoint in data:
                if len(datapoint) == 1 and datapoint[0] not in values:
                    values.append(datapoint[0])
                elif len(datapoint) > 1 and datapoint[feature] not in values:
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
                if len(datapoint) == 1 and datapoint == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(target[dataIndex])
                elif len(datapoint) > 1 and datapoint[feature] == value:
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

        return gain

    def make_tree(self, data, target, featureNames):
        self.num_passes += 1
        if self.num_passes % 100000 == 0:
            print(str(self.num_passes) + " passes")
        # Various initialisations suppressed
        newData = np.array([])
        newClasses = np.array([])
        newNames = np.array([])
        nData = len(data)
        nFeatures = len(featureNames)

        if isinstance(target, str):
            return target

        if len(set(target)) == 1:
            return target[0]

        if nData == 0 or nFeatures == 0 or len(np.unique(data)) == 1:
            # Have reached an empty branch
            if len(target) != 0:
                target_set = set(target)
                frequency = [0] * len(target_set)
                index = 0
                for value in target_set:
                    frequency[index] = np.count_nonzero(target == value)
                    index += 1

                default = target[np.argmax(frequency)]
            else:
                default = self.most_frequent_target()

            return default


        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            values = []
            for feature in range(nFeatures):
                gain[feature] = self.calc_info_gain(data, target, feature)
                # Find possible feature values
                values.extend(self.get_feature_values(data, feature))
            if len(values) > 1:
                values = set(values)
            else:
                values = values[0]

            bestFeature = np.argmin(gain)
            tree = {featureNames[bestFeature]: {}}  # Find the possible feature values
            for value in values:
                index = 0
                # Find the datapoints with each feature value
                for datapoint in data:
                    if datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            datapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature == nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            newDataPoint = datapoint[:bestFeature]
                            newDataPoint = np.append(newDataPoint, datapoint[bestFeature + 1:])
                            datapoint = newDataPoint
                            newNames = featureNames[:bestFeature]
                            newNames = np.hstack((newNames, featureNames[bestFeature + 1:]))

                        if len(newData) == 0:
                            newData = datapoint
                        else:
                            newData = np.vstack((newData, datapoint))

                        if len(newClasses) == 0:
                            newClasses = target[index]
                        else:
                            newClasses = np.append(newClasses, target[index])

                    index += 1
                 # Now recurse to the next level
                subtree = self.make_tree(newData, newClasses, newNames)
                # And on returning, add the subtree on to the tree
                tree[featureNames[bestFeature]][value] = subtree
            return tree


class DecisionTreeModel:
    def __init__(self, tree, feature_names):
        self.tree = tree
        self.model = []
        self.feature_names = feature_names

    def get_node(self, tree, row):
        if isinstance(tree, str):
            return tree

        key = next(iter(tree))
        key_index = np.where(self.feature_names == key)

        node_value = row[key_index][0]
        return self.get_node(tree[key][node_value], row)
        #print("\n\nROW AT KEY INDEX")
        #print(row[key_index])
        #print("KEY - " + str(key))
        #print("NODE - " + str(node_value))
        #if node_value in tree[key]:
        #    print(tree[key][node_value])
        #    print("TREE VALUES?")
        #    print(list(tree.values())[0])
        #    return self.get_node(tree[key][node_value], row)
        #else:
        #    print("\nTREE")
        #    print(tree)
        #    print("TREE KEYS?")
        #    print(tree.keys())
        #    exit(1)

    def predict(self, data):
        for row in data:
            self.model.append(self.get_node(self.tree, row))

        return self.model
