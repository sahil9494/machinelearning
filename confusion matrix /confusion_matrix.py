import csv, sys, numpy as np, pandas as pd


def main():
    confusionMatrixFilename = sys.argv[1]
    df = pd.read_csv(confusionMatrixFilename, sep="\t")
    precisionSet = []
    recallSet = []
    spSet = []
    fdrSet = []
    a = df.values
    # gets rid of the label in the front
    a = np.delete(a, 0, 1)
    calculatePrecisionForEachClass(a.transpose(), precisionSet)
    calculateRecallForEachClass(a.transpose(), recallSet)
    calculateSPForEachClass(a.transpose(), spSet)
    calculateFDRForEachClass(a.transpose(), fdrSet)

    print("Ac", calculateAccuracy(a))
    printFormattedOutput(precisionSet[0], recallSet[0], spSet[0], fdrSet[0])


def printFormattedOutput(precision, recall, sp, fdr):
    classNumber = 1
    print("\tP\t", "R\t", "SP\t", "\tFDR")
    for x in range(len(precision)):
        print("C", classNumber, "\t",precision[x], "\t", recall[x], "\t", sp[x],"\t","\t",fdr[x])
        classNumber += 1


def calculateAccuracy(matrix):
    accuracy = np.sum(matrix.diagonal() / np.sum(matrix))

    return np.round(accuracy, 2)


def calculatePrecisionForEachClass(matrix, precisionSet=None):
    if precisionSet is None:
        precisionSet = []

    precision = np.diag(matrix) / np.sum(matrix, axis=0)
    roundedTo2Decimals = [round(elem, 2) for elem in precision]
    precisionSet.append(roundedTo2Decimals)
    return precisionSet


def calculateRecallForEachClass(matrix, recallSet=None):
    if recallSet is None:
        recallSet = []

    recall = np.diag(matrix) / np.sum(matrix, axis=1)
    roundedTo2Decimals = [round(elem, 2) for elem in recall]
    recallSet.append(roundedTo2Decimals)

    return recallSet


def calculateSPForEachClass(matrix, spSet=None):
    if spSet is None:
        spSet = []

    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)
    TN = matrix.sum() - (FP + FN + TP)

    sp = TN / (FP + TN)
    roundedTo2Decimals = [round(elem, 2) for elem in sp]
    spSet.append(roundedTo2Decimals)

    return spSet


def calculateFDRForEachClass(matrix, fdrSet=None):
    if fdrSet is None:
        fdrSet = []

    FP = matrix.sum(axis=0) - np.diag(matrix)
    TP = np.diag(matrix)

    fdr = FP / (TP + FP)
    roundedTo2Decimals = [round(elem, 2) for elem in fdr]
    fdrSet.append(roundedTo2Decimals)

    return fdrSet


main()
