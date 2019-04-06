
import sys
from builtins import len, enumerate

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split


def main():
    inputLabels = [
        "Feature 1",
        "Feature 2",
        "Feature 3",
        "Feature 4",
        "Feature 5",
        "Feature 6",
        "Feature 7",
        "Feature 8",
        "Feature 9",
        "Feature 10",
        "Feature 11",
        "Feature 12",
        "Feature 13",
        "Feature 14",
        "Feature 15",
        "Feature 16",
        "Feature 17",
        "Feature 18",
        "Feature 19",
        "Feature 20",
        "Feature 21",
        "Feature 22",
        "Feature 23",
        "Feature 24",
        "Feature 25",
        "Feature 26",
        "Feature 27",
        "Feature 28",
        "Feature 29",
        "Feature 30",
        "Feature 31",
        "Feature 32",
        "Feature 33",
        "Feature 34",
        "Feature 35",
        "Feature 36",
        "Feature 37",
        "Feature 38",
        "Feature 39",
        "Feature 40",
        "Feature 41",
        "Feature 42",
        "Feature 43",
        "Feature 44",
        "Feature 45",
        "Feature 46",
        "Feature 47",
        "Feature 48",
        "Feature 49",
        "Feature 50",
        "Feature 51",
        "Feature 52",
        "Feature 53",
        "Feature 54",
        "Feature 55",
        "Feature 56",
        "Feature 57",
        "Feature 58",
        "Feature 59",
        "Feature 60",
        "Feature 61",
        "Feature 62",
        "Feature 63",
        "Feature 64",
        "Feature 65",
        "Feature 66",
        "Feature 67",
        "Feature 68",
        "Feature 69",
        "Feature 70",
        "Feature 71",
        "Y-Values"]

    outputLabels = [
        "Feature 1",
        "Feature 2",
        "Feature 3",
        "Feature 4",
        "Feature 5",
        "Feature 6",
        "Feature 7",
        "Feature 8",
        "Feature 9",
        "Feature 10",
        "Feature 11",
        "Feature 12",
        "Feature 13",
        "Feature 14",
        "Feature 15",
        "Feature 16",
        "Feature 17",
        "Feature 18",
        "Feature 19",
        "Feature 20",
        "Feature 21",
        "Feature 22",
        "Feature 23",
        "Feature 24",
        "Feature 25",
        "Feature 26",
        "Feature 27",
        "Feature 28",
        "Feature 29",
        "Feature 30",
        "Feature 31",
        "Feature 32",
        "Feature 33",
        "Feature 34",
        "Feature 35",
        "Feature 36",
        "Feature 37",
        "Feature 38",
        "Feature 39",
        "Feature 40",
        "Feature 41",
        "Feature 42",
        "Feature 43",
        "Feature 44",
        "Feature 45",
        "Feature 46",
        "Feature 47",
        "Feature 48",
        "Feature 49",
        "Feature 50",
        "Feature 51",
        "Feature 52",
        "Feature 53",
        "Feature 54",
        "Feature 55",
        "Feature 56",
        "Feature 57",
        "Feature 58",
        "Feature 59",
        "Feature 60",
        "Feature 61",
        "Feature 62",
        "Feature 63",
        "Feature 64",
        "Feature 65",
        "Feature 66",
        "Feature 67",
        "Feature 68",
        "Feature 69",
        "Feature 70",
        "Feature 71"]

    trainingDataFileName = sys.argv[1]
    testDataFileName = sys.argv[2]

    trainingData = pd.read_table(trainingDataFileName, header=None)
    trainingData.columns = inputLabels

    testData = pd.read_table(testDataFileName, header=None)
    testData.columns = outputLabels

    X = trainingData.iloc[:, 0:70]  # independent columns
    y = trainingData.iloc[:, -1]  # target column i.e. class
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)

    # best features that will be used in the training data set
    bestFeatures = feat_importances.nlargest(50)
    indices = []

    for index, value in bestFeatures.iteritems():
        indices.append(index)

    bestFeatures.plot(kind='barh')
    import matplotlib.pyplot as plt1

    plt1.show()

    trainingX = trainingData.drop('Y-Values', axis=1).values
    trainingY = trainingData['Y-Values'].values
    from sklearn.neighbors import KNeighborsClassifier

    # Setup arrays to store training and test accuracies
    neighbors = np.arange(5, 10)
    train_accuracy = np.empty(len(neighbors))
    precision = np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    # cross validation to offer training and test data split
    X_train, X_test, y_train, y_test = train_test_split(trainingX, trainingY, test_size=0.21168, stratify=trainingY)

    predictUsingRandomForestClassifier(X_train, y_train, testData)
    predictUsingLogisticRegression(X_train, y_train, testData, y_train)

    # classification for each neighbor
    for i, k in enumerate(neighbors):
        # Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        # Fit the model
        knn.fit(X_train, y_train)

        # Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)
        pred = knn.predict(testData)

        # calculate average precision score for model comparison
        precision[i] = average_precision_score(y_train, pred)

        # Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_train, pred)

        print('prediction for neighbor', i, pred)
        print('precision score for neighbor', i, "->", precision[i])

    precision = np.array(precision)
    averagePrecisionForKnn = precision.mean()
    print("knn average precision", averagePrecisionForKnn)
    # Generate plots for testing and training accuracy
    import matplotlib.pyplot as plt4

    plt4.title('k-NN Varying number of neighbors')
    plt4.plot(neighbors, test_accuracy, label='Testing accuracy')
    plt4.plot(neighbors, train_accuracy, label='Training Accuracy')
    plt4.legend()
    plt4.xlabel('Number of neighbors')
    plt4.ylabel('Accuracy')
    plt4.show()


# use average precision
def plotAUPRCForRandomForest(X_train, y_train, testData, predictions, predictionProbability):
    import matplotlib.pyplot as plt3
    # Create a simple classifier
    classifier = svm.LinearSVC()
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(testData)
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_train, y_score)

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature

    precision, recall, _ = precision_recall_curve(y_train, predictionProbability)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt3.step(recall, precision, color='b', alpha=0.2,
              where='post')
    plt3.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt3.xlabel('Recall')
    plt3.ylabel('Precision')
    plt3.ylim([0.0, 1.05])
    plt3.xlim([0.0, 1.0])
    plt3.title('2-class Precision-Recall(Random Forest) curve: AP= {}'.format(
        average_precision))
    plt3.show()


def writeRandomForestClassifierPredictionsToFile(prediction, predictionProbability):
    predictions = np.array(prediction)
    probablities = np.array(predictionProbability)

    file = open("model-predictions.txt", "w+")
    for i in range(len(prediction)):
        if predictions[i] == 1:
            stringToWrite = "%s \n" % (probablities[i])
            file.write(stringToWrite)


def predictUsingRandomForestClassifier(X_train, y_train, testData):
    from sklearn.ensemble import RandomForestClassifier
    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(testData)
    predictionProbability = clf.predict_proba(testData)
    predictionProbability = [row[0] for row in predictionProbability]

    writeRandomForestClassifierPredictionsToFile(y_pred, predictionProbability)
    print("Precision using random forest classifier:", average_precision_score(y_train, y_pred))

    predictionProbability = np.array(predictionProbability)

    plotAUPRCForRandomForest(X_train, y_train, testData, y_pred, predictionProbability)


def plotAUPRCForLogisticRegression(X_train, y_train, testData, y_pred, predictionProbability):
    import matplotlib.pyplot as plt4
    # Create a simple classifier
    classifier = svm.LinearSVC()
    classifier.fit(X_train, y_train)
    y_score = classifier.decision_function(testData)
    from sklearn.metrics import average_precision_score
    average_precision = average_precision_score(y_train, y_score)

    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    from sklearn.utils.fixes import signature

    precision, recall, _ = precision_recall_curve(y_train, predictionProbability)

    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt4.step(recall, precision, color='b', alpha=0.2,
              where='post')
    plt4.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt4.xlabel('Recall')
    plt4.ylabel('Precision')
    plt4.ylim([0.0, 1.05])
    plt4.xlim([0.0, 1.0])
    plt4.title('2-class Precision-Recall(Logistic Regression) curve: AP= {}'.format(
        average_precision))
    plt4.show()


def predictUsingLogisticRegression(X_train, y_train, testData, y_test):
    # import the class
    from sklearn.linear_model import LogisticRegression
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    # fit the model with data
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(testData)
    predictionProbability = logreg.predict_proba(testData)
    print("Precision using logistic regression classifier:", average_precision_score(y_train, y_pred))
    predictionProbability = [row[0] for row in predictionProbability]
    predictionProbability = np.array(predictionProbability)
    plotAUPRCForLogisticRegression(X_train, y_train, testData, y_pred, predictionProbability)


main()