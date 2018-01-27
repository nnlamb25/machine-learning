from sklearn import datasets
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from HardCodedClassifier import HardCodedClassifier
from kNNClassifier import kNNClassifier


# Prompts the user for their desired data set and returns that data set cleaned up a bit, with its name,
# and whether or not it is a regressor style data set.
def get_data_set():

    while True:
        print("What data would you like to work with?")
        print("1 - Iris data set")
        print("2 - Car Evaluation data set")
        print("3 - Pima Indian Diabetes data set")
        print("4 - Automobile MPG data set")

        data_response = input("> ")

        if data_response == '1':
            return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), \
                   "Iris", False
        elif data_response == '2':
            data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
            columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
            data.columns = columns
            for col in columns:
                data[col] = data[col].astype("category")

            return data, "Car Evaluation", False

        elif data_response == '3':
            data = pd.read_csv(
                "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/"
                "pima-indians-diabetes.data")
            data.columns = ["pregnancies", "glucose", "blood pressure", "tricep thickness", "insulin", "bmi",
                            "pedigree", "age", "diabetic"]
            data["diabetic"].replace([0, 1], ["non-diabetic", "diabetic"], inplace=True)
            data["diabetic"] = data["diabetic"].astype("category")
            data.replace(0, np.NaN, inplace=True)
            data.dropna(inplace=True)
            return data, "Pima Indian Diabetes", False

        elif data_response == '4':
            data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                               delim_whitespace=True)
            data.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight",
                            "acceleration", "year", "origin", "name"]
            new_columns = ["cylinders", "displacement", "horsepower", "weight", "acceleration",
                           "year", "origin", "name", "mpg"]
            for col in new_columns:
                data[col] = data[col].astype("category")
            data = data.reindex(columns=new_columns)
            data.replace("?", np.NaN, inplace=True)
            data.dropna(inplace=True)
            return data, "Automobile MPG", True

        else:
            print("Not a valid input.")


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


# Gets from the user how many times to run the test
def get_number_of_tests():
    is_number = False
    k = 3
    while not is_number or k <= 2:
        print("How many tests do you want to run?")
        if k <= 2:
            print("Must be more than 2 tests.")

        is_number = True
        # Handles error if user inputs a non-integer value
        try:
            k = int(input("> "))
        except:
            print("You must enter a number!")
            is_number = False

    return k


# Gets a single classifier to test data on
def get_classifier(is_regressor_data):
    # Prompts the user for their desired algorithm
    print("Which algorithm would you like to use?")
    # Only returns classifier data that can handle regressors if were using regressor data
    if is_regressor_data:
        while True:
            print("1 - Hard Coded Nearest Neighbor Classifier")
            print("2 - scikit-learn Nearest Neighbor Regressor")
            print("3 - Hard Coded Classifier")
            algorithm_response = input("> ")

            if algorithm_response == '1':
                k = get_k()
                return kNNClassifier(k), "Hard Coded Nearest Neighbor Classifier with a K of " + str(k)
            elif algorithm_response == '2':
                k = get_k()
                return KNeighborsRegressor(n_neighbors=k), "sci-learn Nearest Neighbor Regressor with a K of " + str(k)
            elif algorithm_response == '3':
                return HardCodedClassifier(), "Hard Coded Classifier"
            else:
                print("Not a valid response.")
    else:
        while True:
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
                return KNeighborsClassifier(n_neighbors=k), \
                       "sci-learn Nearest Neighbor Classifier with a K of " + str(k)
            elif algorithm_response == '4':
                return HardCodedClassifier(), "Hard Coded Classifier"
            else:
                print("Not a valid response.")


# Gets a dictionary of classifiers to compare on a data set
def get_multiple_classifiers(is_regressor_data):
    # The classifiers dictionary will have a key with its name as a string
    # and the value will be the classifier class
    classifiers = dict()

    # Gets the classifiers the user wants to compare
    print("Which algorithms would you like to test?")
    print("(Enter an algorithm one at a time, pressing enter after each addition)")
    response = ""

    # If the data set is regressor data, only displays algorithms that can handle it.
    if is_regressor_data:
        while response != "done" and response != "Done":
            print("1 - Hard Coded Nearest Neighbor Classifier")
            print("2 - scikit-learn Nearest Neighbor Regressor")
            print("3 - Hard Coded Classifier")
            print("Type \"done\" when completed.")
            response = input("> ")
            if response == '1':
                k = get_k()
                classifiers["Hard Coded Nearest Neighbor Classifier with a K of " + str(k)] = kNNClassifier(k)
            elif response == '2':
                k = get_k()
                classifiers["sci-learn Nearest Neighbor Regressor with a K of " + str(k)] = KNeighborsRegressor(
                    n_neighbors=k)
            elif response == '3':
                classifiers["Hard Coded Classifier"] = HardCodedClassifier()
            elif response != "Done" and response != "done":
                print("Not a valid response.")

    else:
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
                classifiers["sci-learn Nearest Neighbor Classifier with a K of " + str(k)] = \
                    KNeighborsClassifier(n_neighbors=k)
            elif response == '4':
                classifiers["Hard Coded Classifier"] = HardCodedClassifier()
            elif response != "Done" and response != "done":
                print("Not a valid response.")

    # Returns a map of classifiers with their name as they key and the classifier as the value
    return classifiers


# Cleans up data
def clean(data):
    # Gets the columns in the data which are not numerical values
    # They were given the type "category" when the data was grabbed
    non_numeric_cols = data.select_dtypes(["category"]).columns

    # Replaces all non-numerical values in the data with numerical values
    data[non_numeric_cols] = data[non_numeric_cols].apply(lambda x: x.cat.codes)

    # Sets all the values to their z-score so all values have the same weight
    return data.apply(zscore)


# Tests the algorithm k times using k-cross validation
def k_cross_validation(data_set, algorithm, k):
    kf = KFold(n_splits=k)
    sum_of_accuracies = 0

    # Randomizes the data
    data_set = data_set.sample(frac=1)

    # Splits the data up k times and tests them
    for train, test in kf.split(data_set):
        # Gets the training data
        train = data_set.iloc[train]

        # Gets the testing data
        test = data_set.iloc[test]

        # Gets the accuracy of this test
        accuracy = test_data_set(train, test, algorithm)
        sum_of_accuracies += accuracy

    # Returns the average accuracy
    return sum_of_accuracies / k


# Tests given data on a given algorithm
def test_data_set(train, test, algorithm):

    # Separates the data and puts it into a numpy array
    train_data = np.array(clean(train[train.columns[0: -1]]))
    test_data = np.array(clean(test[test.columns[0: -1]]))
    train_target = np.array(train.iloc[:, -1])
    test_target = np.array(test.iloc[:, -1])

    # Fits the algorithm with data to teach it what to look for
    model = algorithm.fit(train_data, train_target)

    # Tests the algorithm based on what it has been taught
    targets_predicted = model.predict(test_data)

    # Gets the number of correct prediction
    count = 0
    for index in range(len(targets_predicted)):
        if targets_predicted[index] == test_target[index]:
            count += 1

    # Returns the accuracy of the algorithm
    return 0.0 if count == 0 else count / len(targets_predicted)


# Prints the user's given data set
def print_data(data_set):
    # Show the data (the attributes of each instance)
    print("DATA")
    print(data_set)


# Prints the accuracy of a given data set
def print_accuracy(classifier_name, data_set_name, accuracy):
    # Tells the user how accurate their algorithm was on their given data set
    print("\nThe " + classifier_name + " was " + str(round(accuracy * 100, 3)) +
          "% accurate on the " + data_set_name + " Data Set.")


def test_algorithm():
    # Get the data set
    data_set, data_set_name, is_regressor_data = get_data_set()

    # Get the classifier
    classifier, classifier_name = get_classifier(is_regressor_data)

    # Get number of times the user wants to run the classifier on the data
    k = get_number_of_tests()

    # Gets the accuracy of the the algorithm on the data set
    accuracy = k_cross_validation(data_set, classifier, k)

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
    data_set, data_set_name, is_regressor_data = get_data_set()

    # Get the classifiers the user wants to compare
    classifiers = get_multiple_classifiers(is_regressor_data)

    # Get number of times the user wants to run the classifier on the data
    k = get_number_of_tests()

    # Asks the user if they'd like to see their data set
    print("Would you like to see your data? (y, n)")
    see_data_response = input("> ")
    if see_data_response == 'y':
        print_data(data_set)

    # Displays all of the classifier's accuracy
    for classifier_name in classifiers.keys():
        accuracy = k_cross_validation(data_set, classifiers[classifier_name], k)
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
