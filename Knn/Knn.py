import csv, math, sys, operator


def main():
    trainingDataFileName = sys.argv[1]
    testDataFileName = sys.argv[2]

    k = getValueOfKBasedOnScenario()

    trainingSet = []
    testSet = []

    loadTrainingDataSet(trainingDataFileName, trainingSet)
    loadTestDataSet(testDataFileName, testSet)
    arrayOfProbability = []

    for x in range(len(testSet)):
        neighbors = getNearestNeighbors(trainingSet, testSet[x], k)
        result = calculateResultBasedOnClassVotes(neighbors, arrayOfProbability)
        print("'Predicted Class' -> ", result, "'ConditionalProbability' ->", arrayOfProbability[x])


def getValueOfKBasedOnScenario():
    if len(sys.argv) > 3:
        k = int(sys.argv[3])
    else:
        k = 3
    return k


def euclideanDistance(unseenData, seenData, length):
    distance = 0
    for x in range(length):
        distance += pow((unseenData[x] - seenData[x]), 2)
    return math.sqrt(distance)


def loadTrainingDataSet(fileName, trainingDataSet=None):
    if trainingDataSet is None:
        trainingDataSet = []
    with open(fileName, 'r') as trainingData:
        lines = csv.reader(trainingData, delimiter='\t')
        dataset = list(lines)
        for x in range(1, len(dataset)):
            for y in range(9):
                dataset[x][y] = float(dataset[x][y])
            trainingDataSet.append(dataset[x])

    return trainingDataSet


def loadTestDataSet(fileName, testDataSet=None):
    if testDataSet is None:
        testDataSet = []
    with open(fileName, 'r') as testData:
        lines = csv.reader(testData, delimiter='\t')
        dataset = list(lines)
        for x in range(1, len(dataset)):
            for y in range(9):
                dataset[x][y] = float(dataset[x][y])
            testDataSet.append(dataset[x])

    return testDataSet


def getNearestNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))

    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def conditionalProbability(calculatedResult, resultsInNearestNeighbors):
    counter = 0

    for x in range(len(resultsInNearestNeighbors)):
        if resultsInNearestNeighbors[x] == calculatedResult:
            counter += 1

    return counter / len(resultsInNearestNeighbors)


def calculateResultBasedOnClassVotes(neighbors, arrayOfProbability):
    classVotes = {}
    classForNearestNeighbors = []
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        classForNearestNeighbors.append(neighbors[x][9])
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    prob = conditionalProbability(sortedVotes[0][0], classForNearestNeighbors)
    arrayOfProbability.append(prob)

    return sortedVotes[0][0]


main()
