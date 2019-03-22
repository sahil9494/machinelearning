import csv
import numpy as np
import pandas as pd
import sys
from random import randrange

import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier


def main():
    trainingDataFileName = sys.argv[1]

    nameArray = [
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
        "Feature 72",
        "Feature 73",
        "Feature 74",
        "Feature 75",
        "Feature 76",
        "Feature 77",
        "Feature 78",
        "Feature 79",
        "Feature 80",
        "Feature 81",
        "Feature 82",
        "Feature 83",
        "Feature 84",
        "Feature 85",
        "Feature 86",
        "Feature 87",
        "Feature 88",
        "Feature 89",
        "Feature 90",
        "Feature 91",
        "Feature 92",
        "Feature 93",
        "Feature 94",
        "Feature 95",
        "Feature 96",
        "Feature 97",
        "Feature 98",
        "Feature 99",
        "Feature 100",
        "Feature 101",
        "Feature 102",
        "Feature 103",
        "Feature 104",
        "Feature 105",
        "Feature 106",
        "Feature 107",
        "Feature 108",
        "Feature 109",
        "Feature 110",
        "Feature 111",
        "Feature 112",
        "Feature 113",
        "Feature 114",
        "Feature 115",
        "Feature 116",
        "Feature 117",
        "Feature 118",
        "Feature 119",
        "Feature 120",
        "Feature 121",
        "Feature 122",
        "Feature 123",
        "Feature 124",
        "Feature 125",
        "Feature 126",
        "Feature 127",
        "Feature 128",
        "Feature 129",
        "Feature 130",
        "Feature 131",
        "Feature 132",
        "Feature 133",
        "Feature 134",
        "Feature 135",
        "Feature 136",
        "Feature 137",
        "Feature 138",
        "Feature 139",
        "Feature 140",
        "Feature 141",
        "Feature 142",
        "Feature 143",
        "Feature 144",
        "Feature 145",
        "Feature 146",
        "Feature 147",
        "Feature 148",
        "Feature 149",
        "Feature 150",
        "Feature 151",
        "Feature 152",
        "Feature 153",
        "Feature 154",
        "Feature 155",
        "Feature 156",
        "Feature 157",
        "Feature 158",
        "Feature 159",
        "Feature 160",
        "Feature 161",
        "Feature 162",
        "Feature 163",
        "Feature 164",
        "Feature 165",
        "Feature 166",
        "Feature 167",
        "Feature 168",
        "Feature 169",
        "Feature 170",
        "Feature 171",
        "Feature 172",
        "Feature 173",
        "Feature 174",
        "Feature 175",
        "Feature 176",
        "Feature 177",
        "Feature 178",
        "Feature 179",
        "Feature 180",
        "Feature 181",
        "Feature 182",
        "Feature 183",
        "Feature 184",
        "Feature 185",
        "Feature 186",
        "Feature 187",
        "Feature 188",
        "Feature 189",
        "Feature 190",
        "Feature 191",
        "Feature 192",
        "Feature 193",
        "Feature 194",
        "Feature 195",
        "Feature 196",
        "Feature 197",
        "Feature 198",
        "Feature 199",
        "Feature 200",
        "Feature 201",
        "Feature 202",
        "Feature 203",
        "Feature 204",
        "Feature 205",
        "Feature 206",
        "Feature 207",
        "Feature 208",
        "Feature 209",
        "Feature 210",
        "Feature 211",
        "Feature 212",
        "Feature 213",
        "Feature 214",
        "Feature 215",
        "Feature 216",
        "Feature 217",
        "Feature 218",
        "Feature 219",
        "Feature 220",
        "Feature 221",
        "Feature 222",
        "Feature 223",
        "Feature 224",
        "Feature 225",
        "Feature 226",
        "Feature 227",
        "Feature 228",
        "Feature 229",
        "Feature 230",
        "Feature 231",
        "Feature 232",
        "Feature 233",
        "Feature 234",
        "Feature 235",
        "Feature 236",
        "Feature 237",
        "Feature 238",
        "Feature 239",
        "Feature 240",
        "Feature 241",
        "Feature 242",
        "Feature 243",
        "Feature 244",
        "Feature 245",
        "Feature 246",
        "Feature 247",
        "Feature 248",
        "Feature 249",
        "Feature 250",
        "Feature 251",
        "Feature 252",
        "Feature 253",
        "Feature 254",
        "Feature 255",
        "Feature 256",
        "Feature 257",
        "Feature 258",
        "Feature 259",
        "Feature 260",
        "Feature 261",
        "Feature 262",
        "Feature 263",
        "Feature 264",
        "Feature 265",
        "Feature 266",
        "Feature 267",
        "Feature 268",
        "Feature 269",
        "Feature 270",
        "Feature 271",
        "Feature 272",
        "Feature 273",
        "Feature 274",
        "Feature 275",
        "Feature 276",
        "Feature 277",
        "Feature 278",
        "Feature 279",
        "Feature 280",
        "Feature 281",
        "Feature 282",
        "Feature 283",
        "Feature 284",
        "Feature 285",
        "Feature 286",
        "Feature 287",
        "Feature 288",
        "Feature 289",
        "Feature 290",
        "Feature 291",
        "Feature 292",
        "Feature 293",
        "Feature 294",
        "Feature 295",
        "Feature 296",
        "Feature 297",
        "Feature 298",
        "Feature 299",
        "Feature 300",
        "Feature 301",
        "Feature 302",
        "Feature 303",
        "Feature 304",
        "Feature 305",
        "Feature 306",
        "Feature 307",
        "Feature 308",
        "Feature 309",
        "Feature 310",
        "Feature 311",
        "Feature 312",
        "Feature 313",
        "Feature 314",
        "Feature 315",
        "Feature 316",
        "Feature 317",
        "Feature 318",
        "Feature 319",
        "Feature 320",
        "Feature 321",
        "Feature 322",
        "Feature 323",
        "Feature 324",
        "Feature 325",
        "Feature 326",
        "Feature 327",
        "Feature 328",
        "Feature 329",
        "Feature 330",
        "Feature 331",
        "Feature 332",
        "Feature 333",
        "Feature 334",
        "Feature 335",
        "Feature 336",
        "Feature 337",
        "Feature 338",
        "Feature 339",
        "Feature 340",
        "Feature 341",
        "Feature 342",
        "Feature 343",
        "Feature 344",
        "Feature 345",
        "Feature 346",
        "Feature 347",
        "Y-Values"]

    trainingData = pd.read_table(trainingDataFileName, header=None)
    trainingData.columns = nameArray

    X = trainingData.iloc[:, 0:347]  # independent columns
    y = trainingData.iloc[:, -1]  # target column i.e. class
    model = ExtraTreesClassifier()
    model.fit(X, y)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    bestFeatures = feat_importances.nlargest(50)
    indices = []

    for index, value in bestFeatures.iteritems():
        indices.append(index)

    indices.append("Y-Values")

    df = trainingData[indices]
    bestFeatures.plot(kind='barh')
    plt.show()

    df = df.reset_index()
    folds = cross_validation_split(df.values, 5)

    # compute for each fold
    i = 0
    for fold in folds:
        train_set = list(folds)

        # remove the current fold from the training set
        train_set = removearray(train_set, fold)
        train_set = sum(train_set, [])

        # initialize an empty list to put test data
        test_set = list()

        for row in fold:
            row_copy = list(row)
            # make the test data
            test_set.append(row_copy)

            # make the last column None as we remove the class columns from the test set
            # to use it for prediction. This is the column that our model is going to
            # predict for us.
            row_copy[-1] = None

        # make a prediction using sci-kit learn knn classifier for each variable (50 best selected earlier using
        # extra trees classifier method)
        i = i + 1
        for variable in range(len(indices) - 1):
            print("values for fold", i, "feature", variable + 1)
            x_train = getXTrainingValueForEachVariable(train_set, variable)
            targetTrainingY = getTargetYValues(train_set)
            x_test = getXTestingValueForEachVariable(test_set, variable)
            trueYForFold = getTrueY(fold)

            # range of the number of neighbors
            k_range = range(1, 11)
            scores = {}

            # A list to keep track of the accuracy scores of the model
            scores_list = []

            # A list to keep track of the best K, based on the highest accuracy(first-occurrence)
            listOfHighestK = []

            for k in k_range:
                knn = KNeighborsClassifier(k)

                # fit the model on the training x and y (y being the last column- 51 in our case)
                knn.fit(x_train, targetTrainingY)
                # perform predictions on the test data
                predictedY = knn.predict(x_test)

                # score the predictions of the model
                scores[k] = metrics.accuracy_score(trueYForFold, predictedY)
                scores_list.append(scores[k])

            bestValueForK = getIndexOfHighestValue(scores_list)
            listOfHighestK.append(bestValueForK)

            # plots the accuracy for each feature on each fold with different
            # number of neighbors
            plt.title('Accuracy for the Feature')
            plt.plot(scores_list, label='Testing Accuracy')
            plt.legend()
            plt.xlabel('Number of neighbors')
            plt.ylabel('Classification Accuracy in %')
            plt.show()

            # calculates average of each row of scores we get for each neighbor (/10)
            highestScore = getHighestScoreFromEachNeighborsIndividualScore(scores_list)
            optimalNumberOfNeighbors = getAverageForEachNeighbor(listOfHighestK)

            print("Highest accuracy score: ", highestScore * 100, "%")
            print("Optimal number of neighbors: ", optimalNumberOfNeighbors)

        # calculate residual sum of squares
        # residualScore = residualSumOfSquares(predictedY, trueYForFold)


def getHighestScoreFromEachNeighborsIndividualScore(scores_list):
    index = np.array(scores_list).argmax()
    return scores_list[index]


def getAverageForEachNeighbor(listOfHighestK):
    return np.mean(listOfHighestK).astype(int)


def getIndexOfHighestValue(scores_list):
    max_value = np.array(scores_list)
    max_value = max_value.astype(float)
    max_value = max_value.argmax()
    return max_value


# supports removal of np array from another np array
def removearray(L, arr):
    ind = 0
    size = len(L)
    while ind != size and not np.array_equal(L[ind], arr):
        ind += 1
    if ind != size:
        L.pop(ind)
    else:
        raise ValueError('array not found in list.')

    return L


def getXTestingValueForEachVariable(fold, variable):
    tDash = [item[variable] for item in fold]
    tDash = np.array(tDash)
    tDash = tDash.astype(float)
    c = tDash.reshape(-1, 1)
    return c


def getXTrainingValueForEachVariable(train_set, variable):
    lDash = [item[variable] for item in train_set]
    lDash = np.array(lDash)
    b = lDash.reshape(-1, 1)
    return b


def getTrueY(fold):
    trueYForFold = [item[-1] for item in fold]
    trueYForFold = np.array(trueYForFold)
    trueYForFold = trueYForFold.astype(int)
    return trueYForFold


def getTargetYValues(train_set):
    targetTrainingY = [item[-1] for item in train_set]
    targetTrainingY = np.array(targetTrainingY)
    targetTrainingY = targetTrainingY.astype(int)
    return targetTrainingY


def residualSumOfSquares(predictions, targets):
    differences = predictions - targets
    differences_squared = differences ** 2
    mean_of_differences_squared = differences_squared.mean()
    rmse_val = np.sqrt(mean_of_differences_squared)

    return rmse_val


# Split a dataset into k folds
def cross_validation_split(dataset, folds=3):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / folds)
    for i in range(folds):  # first loop of grid search
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))  # set L without the ith fold
        dataset_split.append(fold)

    return dataset_split


def loadTrainingDataSet(fileName, trainingDataSet=None):
    if trainingDataSet is None:
        trainingDataSet = []
    with open(fileName, 'r') as trainingData:
        lines = csv.reader(trainingData, delimiter='\t')
        dataset = list(lines)
        for x in range(0, len(dataset)):
            for y in range(0, 348):
                dataset[x][y] = float(dataset[x][y])
            trainingDataSet.append(dataset[x])

    return trainingDataSet


main()

