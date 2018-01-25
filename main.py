from sklearn import datasets
import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from HardCodedClassifier import HardCodedClassifier
from kNNClassifier import kNNClassifier


# Prompts the user for their desired data set
def get_data_set():
    print("What data would you like to work with?")
    print("1 - Iris data set")
    print("2 - Car Evaluation data set")
    print("3 - Pima Indian Diabetes data set")
    print("4 - Automobile MPG data set")
    data_response = input("> ")

    if data_response == '1':
        return pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"), "Iris"
    elif data_response == '2':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data")
        columns = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
        data.columns = columns
        for col in columns:
            data[col] = data[col].astype("category")

        return data, "Car Evaluation"
    elif data_response == '3':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data")
        data.columns = ["pregnancies", "glucose", "blood pressure", "tricep thickness", "insulin", "bmi", "pedigree", "age", "diabetic"]
        data["diabetic"].replace([0, 1], ["non-diabetic", "diabetic"], inplace=True)
        data["diabetic"] = data["diabetic"].astype("category")
        data.replace(0, np.NaN, inplace=True)
        data.dropna(inplace=True)
        return data, "Pima Indian Diabetes"
    elif data_response == '4':
        data = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data",
                           delim_whitespace=True)
        data.columns = ["mpg", "cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin", "name"]
        new_columns = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "year", "origin", "name", "mpg"]
        for col in new_columns:
            data[col] = data[col].astype("category")
        data = data.reindex(columns=new_columns)
        data.replace("?", np.NaN, inplace=True)
        data.dropna(inplace=True)
        return data, "Automobile MPG"
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
    while True:
        print("1 - scikit-learn Gaussian")
        print("2 - Hard Coded Nearest Neighbor Classifier")
        print("3 - scikit-learn Nearest Neighbor Classifier")
        print("4 - scikit-learn Nearest Neighbor Regressor")
        print("5 - Hard Coded Classifier")
        algorithm_response = input("> ")

        if algorithm_response == '1':
            return GaussianNB(), "Gaussian"
        elif algorithm_response == '2':
            k = get_k()
            return kNNClassifier(k), "Hard Coded Nearest Neighbor Classifier with a K of " + str(k)
        elif algorithm_response == '3':
            k = get_k()
            return KNeighborsClassifier(n_neighbors=k), "sci-learn Nearest Neighbor Classifier with a K of " + str(k)
        elif algorithm_response == '4':
            k = get_k()
            return KNeighborsRegressor(n_neighbors=k), "sci-learn Nearest Neighbor Regressor with a K of " + str(k)
        elif algorithm_response == '5':
            return HardCodedClassifier(), "Hard Coded Classifier"
        else:
            print("Not a valid response.")


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
        print("4 - scikit-learn Nearest Neighbor Regressor")
        print("5 - Hard Coded Classifier")
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
            k = get_k()
            classifiers["sci-learn Nearest Neighbor Regressor with a K of " + str(k)] = KNeighborsRegressor(n_neighbors=k)
        elif response == '5':
            classifiers["Hard Coded Classifier"] = HardCodedClassifier()
        elif response != "Done" and response != "done":
            print("Not a valid response.")

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


# Tests given data on a given algorithm
def test_data_set(data_set, algorithm):
    # Randomizes the data set to prepare to split between teaching and testing
    data_set = data_set.sample(frac=1).reset_index(drop=True)

    # Grabs only the data from the data set (all columns but the last)
    data = data_set[data_set.columns[0: -1]]

    # Cleans the data of unweighted data and empty values
    data = clean(data)

    # Grabs the target from the data set (last column)
    target = data_set.iloc[:, -1]

    # Splits the data into the train data, train targets, test data, and test targets
    train_data = data[:int(0.7 * len(data))]
    test_data = data[int(0.7 * len(data)) + 1:]
    train_target = target[:int(0.7 * len(target))]
    test_target = target[int(0.7 * len(target)) + 1:]

    # Fits the algorithm with the data (training the algorithm)
    model = algorithm.fit(train_data, train_target)

    # Tests the algorithm now that it has been trained
    targets_predicted = model.predict(test_data)

    # Finds how many tests the algorithm correctly predicted
    count = 0
    for index in range(len(targets_predicted)):
        if targets_predicted[index] == test_target.iloc[index]:
            count += 1

    # Returns the accuracy of the algorithm
    return 0.0 if count == 0 else count / len(targets_predicted)


# Prints the user's given data set
def print_data(data_set):
    # Show the data (the attributes of each instance)
    print("DATA")
    print(data_set)
    #print(data_set.data)

    # Show the target values (in numeric format) of each instance
    #print("\nTARGET VALUES:")
    #print(data_set.target)

    # Show the actual target names that correspond to each number
    #print("\nTARGET NAMES:")
    #print(data_set.target_names)


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
