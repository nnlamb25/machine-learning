from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from HardCodedClassifier import HardCodedClassifier


# Prompts the user for their desired dataset
def get_data_set():
    print("What data would you like to work with?")
    print("1 - Iris data set")
    data_response = input("> ")

    if data_response == '1':
        return datasets.load_iris(), "Iris"
    else:
        return datasets.load_iris(), "Iris"


def get_classifier():
    # Prompts the user for their desired algorithm
    print("Which algorithm would you like to use?")
    print("1 - Gaussian")
    print("2 - Hard Coded Classifier")
    algorithm_response = input("> ")

    if algorithm_response == '1':
        return GaussianNB(), "Gaussian"
    elif algorithm_response == '2':
        return HardCodedClassifier(), "Hard Coded Classifier"
    else:
        return HardCodedClassifier(), "Hard Coded Classifier"


# Tests given data on a given algorithm
def test_data_set(data_set, algorithm):
    # Splits the data randomly, gives 70% to train and 30% to test
    train_data, test_data, train_target, test_target = \
        train_test_split(data_set.data, data_set.target, test_size=0.3)

    # Fits the algorithm with the data (training the algorithm)
    model = algorithm.fit(train_data, train_target)

    # Tests the algorithm now that it has been trained
    targets_predicted = model.predict(test_data)

    # Finds how many tests the algorithm correctly predicted
    count = 0
    for index in range(len(targets_predicted)):
        if targets_predicted[index] == test_target[index]:
            count += 1

    # Returns the accuracy of the algorithm
    return count / len(targets_predicted)


# Prints the user's given dataset
def print_data(dataset):
    # Show the data (the attributes of each instance)
    print("DATA:")
    print(dataset.data)

    # Show the target values (in numeric format) of each instance
    print("\nTARGET VALUES:")
    print(dataset.target)

    # Show the actual target names that correspond to each number
    print("\nTARGET NAMES:")
    print(dataset.target_names)


def main():
    # Get the data set
    data_set, data_set_name = get_data_set()

    # Get the classifier
    classifier, classifier_name = get_classifier()

    # Gets the accuracy of the the algorithm on the data set
    accuracy = test_data_set(data_set, classifier)

    # Prompts the user if they would like to see their data set
    print("Would you like to see your data? (y, n)")
    see_data_response = input("> ")

    if see_data_response == 'y':
        print_data(data_set)

    # Tells the user how accurate their algorithm was on their given data set
    print("\nThe " + classifier_name + " algorithm was " + str(round(accuracy * 100, 3)) +
          "% accurate on the " + data_set_name + " data set.")


main()
