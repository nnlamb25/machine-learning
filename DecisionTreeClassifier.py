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
        print("\n\n\nENDING TREE:")
        print(tree)

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
                elif len(datapoint) > 1 and datapoint[feature] not in values: #elif datapoint[feature] not in values:
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
                #if datapoint.size == 1 and datapoint == value:
                #    featureCounts[valueIndex] += 1
                #    newClasses.append(target[dataIndex])
                if len(datapoint) == 1 and datapoint == value:
                    featureCounts[valueIndex] += 1
                    newClasses.append(target[dataIndex])
                elif len(datapoint) > 1 and datapoint[feature] == value:#elif datapoint.size > 1 and datapoint[feature] == value:
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
        newData = np.array([])
        newClasses = np.array([])#newTarget = np.array([])
        newNames = np.array([])
        nData = len(data)
        nFeatures = len(featureNames)

        if isinstance(target, str):
            return target

        if nData == 0 or nFeatures == 0:
            # Have reached an empty branch
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

        elif len(target[0]) == nData:
            # Only 1 class remains
            return target[0]
        else:
            # Choose which feature is best
            gain = np.zeros(nFeatures)
            values = []
            for feature in range(nFeatures):
                g = self.calc_info_gain(data, target, feature)
                gain[feature] = np.argmin(g)
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
                            datapoint = datapoint[:bestFeature]
                            datapoint.extend(datapoint[bestFeature + 1:])
                            newNames = featureNames[:bestFeature]
                            newNames.extend(featureNames[bestFeature + 1:])

                        if len(newData) == 0:
                            newData = datapoint
                        else:
                            newData = np.vstack([newData, datapoint])

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

        # If there is no more data and no more features, return the most frequent value
    """    if nData == 0 and nFeatures == 0:
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

            if gain.size > 0:
                bestFeature = np.argmin(gain)
            else:
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

            tree = {featureNames[bestFeature]: {}}

            # Find where those values appear in data[feature] and the corresponding class
            for value in values:
                index = 0
                # Find the datapoints with each feature value
                for datapoint in data:
                    if len(datapoint) == 1 and datapoint == value:
                        if bestFeature == 0:
                            #datapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature == nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            np.append(datapoint, datapoint[bestFeature + 1:])#datapoint.extend(datapoint[bestFeature + 1:])
                            newNames = featureNames[:bestFeature]
                            np.append(newNames, featureNames[bestFeature + 1:])#newNames.extend(featureNames[bestFeature + 1:])

                        if len(newData) == 0 and len(datapoint) > 0:
                            newData = datapoint
                        elif len(datapoint) > 0:
                            np.append(newData, datapoint)#newData.append(datapoint)

                        if len(newTarget) == 0 and len(target[index]) > 0:
                            newTarget = target[index]
                        elif len(target[index]) > 0:
                            np.append(newTarget, target[index])#newTarget.append(target[index])

                    elif len(datapoint) > 0 and datapoint[bestFeature] == value:
                        if bestFeature == 0:
                            datapoint = datapoint[1:]
                            newNames = featureNames[1:]
                        elif bestFeature == nFeatures:
                            datapoint = datapoint[:-1]
                            newNames = featureNames[:-1]
                        else:
                            datapoint = datapoint[:bestFeature]
                            np.append(datapoint, datapoint[bestFeature + 1:])#datapoint.extend(datapoint[bestFeature + 1:])
                            newNames = featureNames[:bestFeature]
                            np.append(newNames, featureNames[bestFeature + 1:])#newNames.extend(featureNames[bestFeature + 1:])

                        if len(newData) == 0 and len(datapoint) > 0:
                            newData = datapoint
                        elif len(datapoint) > 0:
                            np.append(newData, datapoint)#newData.append(datapoint)

                        if len(newTarget) == 0 and len(target[index]) > 0:
                            newTarget = target[index]
                        elif len(target[index]) > 0:
                            np.append(newTarget, target[index])#newTarget.append(target[index])

                    index += 1
                # Now recurse to the next level
                subtree = self.make_tree(newData, newTarget, newNames)
                # And on returning, add the subtree on to the tree
                tree[featureNames[bestFeature]][value] = subtree
            return tree
"""

class DecisionTreeModel:
    def __init__(self, tree, feature_names):
        self.tree = tree
        self.model = []
        self.feature_names = feature_names

    def get_feature_index(self, feature):
        return np.where(self.feature_names == feature)

    def get_node(self, tree, row):
        #print("\nTREE")
        #print(tree)
        #print("LENGTH")
        #print(sum(len(v) for v in tree.items()))
        #if sum(len(v) for v in tree.items()) > 1:
        key = next(iter(tree))
        key_index = self.get_feature_index(key)
        node_value = row[key_index][0]
        #print("\nKEY INDEX:")
        #print(key_index)
        #print("\nKEY:")
        #print(key)
        #print("\nNODE VALUE:")
        #print(node_value)
        #print()
        #print(tree[key][node_value])
        return tree[key][node_value]
        #return self.get_node(tree[key][node_value], row)
        #else:
            #return 'r'

    def predict(self, data):
        for row in data:
            self.model.append(self.get_node(self.tree, row))
        #for row in data:
         #   for key in self.tree:
         #       key_index = self.get_feature_index(key)
         #       print(row)
         #       print(key)
         #       print(key_index)
                #print(row['physician fee'])
         #       print(row[key_index])
          #  self.model.append('r')
        #unique, pos = np.unique(self.target, return_inverse=True)
        #counts = np.bincount(pos)
        #maxpos = counts.argmax()
        #most_common_target = self.target[maxpos]
        #for _ in data:
        #    self.model.append(most_common_target)

        return self.model
