from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from HardCodedClassifier import HardCodedClassifier
from kNNClassifier import kNNClassifier


# Prompts the user for their desired data set
def get_data_set():
    print("What data would you like to work with?")
    print("1 - Iris data set")
    data_response = input("> ")

    if data_response == '1':
        return datasets.load_iris(), "Iris"
    else:
        return datasets.load_iris(), "Iris"


# Gets the user's desired value of K for the nearest neighbor algorithm
def get_k():
    print("What value would you like to use for K?")
    is_number = False
    k = 1
    while not is_number or k <= 0:
        if k <= 0:
            print("Value must be larger than 0")

        is_number = True
        # Handles error if user inputs a non-integer value
        try:
            k = int(input("> "))
        except:
            print("You must enter a number!")
            is_number = False

    return k


# Gets a single classifier to test data on
def get_classifier():
    # Prompts the user for their desired algorithm
    print("Which algorithm would you like to use?")
    print("1 - scikit-learn Gaussian")
    print("2 - Hard Coded Nearest Neighbor Classifier")
    print("3 - scikit-learn Nearest Neighbor Classifier")
    print("4 - Hard Coded Classifier")
    algorithm_response = input("> ")

    if algorithm_response == '1':
        return GaussianNB(), "Gaussian"
    elif algorithm_response == '2':
        k = get_k()
        return kNNClassifier(k), "Hard Coded Nearest Neighbor Classifier with a K of " + str(k)
    elif algorithm_response == '3':
        k = get_k()
        return KNeighborsClassifier(n_neighbors=k), "sci-learn Nearest Neighbor Classifier with a K of " + str(k)
    else:
        return HardCodedClassifier(), "Hard Coded Classifier"


# Gets a dictionary of classifiers to compare on a data set
def get_multiple_classifiers():
    # The classifiers dictionary will have a key with its name as a string
    # and the value will be the classifier class
    classifiers = dict()

    # Gets the classifiers the user wants to compare
    print("Which algorithms would you like to test?")
    print("(Enter an algorithm one at a time, pressing enter after each addition)")
    response = ""
    while response != "done" and response != "Done":
        print("1 - scikit-learn Gaussian")
        print("2 - Hard Coded Nearest Neighbor Classifier")
        print("3 - scikit-learn Nearest Neighbor Classifier")
        print("4 - Hard Coded Classifier")
        print("Type \"done\" when completed.")
        response = input("> ")
        if response == '1':
            classifiers["scikit-learn Gaussian"] = GaussianNB()
        elif response == '2':
            k = get_k()
            classifiers["Hard Coded Nearest Neighbor Classifier with a K of " + str(k)] = kNNClassifier(k)
        elif response == '3':
            k = get_k()
            classifiers["sci-learn Nearest Neighbor Classifier with a K of " + str(k)] = KNeighborsClassifier(n_neighbors=k)
        elif response == '4':
            classifiers["Hard Coded Classifier"] = HardCodedClassifier()
        elif response != "Done" and response != "done":
            print("Not a valid response.")

    return classifiers


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


# Prints the user's given data set
def print_data(data_set):
    # Show the data (the attributes of each instance)
    print("DATA:")
    print(data_set.data)

    # Show the target values (in numeric format) of each instance
    print("\nTARGET VALUES:")
    print(data_set.target)

    # Show the actual target names that correspond to each number
    print("\nTARGET NAMES:")
    print(data_set.target_names)


# Prints the accuracy of a given data set
def print_accuracy(classifier_name, data_set_name, accuracy):
    # Tells the user how accurate their algorithm was on their given data set
    print("\nThe " + classifier_name + " was " + str(round(accuracy * 100, 3)) +
          "% accurate on the " + data_set_name + " Data Set.")


def test_algorithm():
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

    # Displays the classifier's accuracy
    print_accuracy(classifier_name, data_set_name, accuracy)


# Compares multiple user given classifiers
def compare_algorithms():
    # Get the data set
    data_set, data_set_name = get_data_set()

    # Get the classifiers the user wants to compare
    classifiers = get_multiple_classifiers()

    print("Would you like to see your data? (y, n)")
    see_data_response = input("> ")
    if see_data_response == 'y':
        print_data(data_set)

    # Displays all of the classifier's accuracy
    for classifier_name in classifiers.keys():
        accuracy = test_data_set(data_set, classifiers[classifier_name])
        print_accuracy(classifier_name, data_set_name, accuracy)


# Asks the user if they would like to test an algorithm or compare many algorithms
def main():
    print("What would you like to do?")
    print("1 - Test an algorithm")
    print("2 - Compare multiple algorithms")
    job_response = input("> ")

    if job_response == '2':
        compare_algorithms()
    else:
        test_algorithm()


# Runs the program
main()
