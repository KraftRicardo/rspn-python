import numpy as np
import scipy.stats
import time

from scripts.main import readCsvAsNumpyArray, readCsvAsNumpyArrayAndHeader

RAND_STATE = 1337


def make_matrix(data):
    if data.ndim == 1:
        data = data[:, np.newaxis]
    else:
        assert data.ndim == 2, "Data must be 2 dimensional {}".format(data.shape)

    return data


def empirical_copula_transformation(data):
    ones_column = np.ones((data.shape[0], 1))
    data = np.concatenate((np.apply_along_axis(ecdf, 0, data), ones_column), axis=1)
    return data


def ecdf(X):
    return scipy.stats.rankdata(X, method='max') / len(X)


# For: def rdc_transformer(data_slice, k=None, s=1. / 6., non_linearity=np.sin, return_matrix=False, ohe=True, rand_gen=None):
def rdc_transformer(data, rows, cols, gaussian_path, k, s, non_linearity, rand_gen, return_matrix):
    # print("Data\n", data)
    # print("Rows\n", rows)
    # print("Cols\n", cols)

    # features = [data_slice.getFeatureData(f) for f in data_slice.cols]
    features = [getFeatureData(data, rows, f) for f in cols]

    # forcing two columness
    features = [make_matrix(f) for f in features]

    # transform through the empirical copula
    features = [empirical_copula_transformation(f) for f in features]

    # random projection through a gaussian
    if (gaussian_path != None):  # e.g. "res/t3/gaussian_0"
        random_gaussians = [readCsvAsNumpyArray(gaussian_path) for f in features]
    else:
        random_gaussians = [rand_gen.normal(size=(f.shape[1], k)) for f in features]

    rand_proj_features = [s / f.shape[1] * np.dot(f, N) for f, N in zip(features, random_gaussians)]

    nl_rand_proj_features = [non_linearity(f) for f in rand_proj_features]

    # apply non-linearity
    if return_matrix:
        return np.concatenate(nl_rand_proj_features, axis=1)
    else:
        # print([f.shape for f in nl_rand_proj_features])
        return [np.concatenate((f, np.ones((f.shape[0], 1))), axis=1) for f in nl_rand_proj_features]


def getFeatureData(data, rows, f):
    return data[rows, :][:, f]  # original


# small version of the data slice class
def generate_normal_distributed_data(numberOfRows: int, numberOfColumns: int, mean: float, standardDeviation: float):
    rand_gen = np.random.RandomState(1337)  # set seed for random generator

    matrix = (numberOfRows, numberOfColumns)
    data = rand_gen.normal(size=matrix, loc=mean, scale=standardDeviation)
    rows = np.arange(numberOfRows)
    cols = np.arange(numberOfColumns)

    return data, rows, cols


def measure_time(filename):
    # standard param
    k = 20
    s = 1. / 6.
    non_linearity = np.sin
    rand_gen = np.random.RandomState(RAND_STATE)
    return_matrix = False

    data, _ = readCsvAsNumpyArrayAndHeader(filename)
    rows = np.arange(data.shape[0])
    cols = np.arange(data.shape[1])

    startTime = time.time()
    for i in range(100):
        matrix = rdc_transformer(data, rows, cols, None, k, s, non_linearity, rand_gen, return_matrix)
    endTime = time.time()
    print("Time=", (endTime - startTime))


def performance_test():
    for i in range(1000001):
        if i == 1000000:
            print(i)

    nRows = 0
    for i in range(10):
        nRows += 10000
        print("10x" + str(nRows), end=" ")
        measure_time("res/P-Test/10x" + str(nRows) + ".csv")

    nCols = 0
    for i in range(18):
        nCols += 5
        print(str(nCols) + "x10000", end=" ")
        measure_time("res/P-Test/" + str(nCols) + "x10000.csv")


if __name__ == '__main__':
    # for timing:
    performance_test()

    # for value comparison:
    # filename = "res/t4/t4.csv";
    # gaussian_path = "res/t4/gaussian_0";
    # filename = "res/t3/t3.csv";
    # gaussian_path = "res/t3/gaussian_0";
    # matrix = rdc_transformer(data, rows, cols, gaussian_path, k, s, non_linearity, rand_gen, return_matrix);
