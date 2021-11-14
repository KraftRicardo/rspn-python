import itertools
import numpy as np
from sklearn.cross_decomposition import CCA

from scripts.main import readCsvAsNumpyArrayAndHeader
from testrun.rdc_transformer_copy import rdc_transformer, RAND_STATE

GLOBAL_RDC_FEATURES = []


def rdc_test(rdc_features, cols, n_jobs=7):
    n_features = len(cols)

    if n_jobs is None:
        n_jobs = n_features * n_features

    # build adjacency matrix
    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    GLOBAL_RDC_FEATURES.clear()
    GLOBAL_RDC_FEATURES.extend(rdc_features)

    pairwise_comparisons = itertools.combinations(np.arange(n_features), 2)
    rdc_vals = None
    # with Pool(n_jobs) as p:
    # p = Pool(n_jobs)

    cca = CCA(n_components=1)
    from joblib import Parallel, delayed
    rdc_vals = Parallel(n_jobs=n_jobs)(delayed(rdc_cca)((i, j, cca)) for i, j in pairwise_comparisons)

    # with concurrent.futures.ProcessPoolExecutor(n_jobs) as p:
    #     rdc_vals = p.map(rdc_cca, [(i, j) for i, j in pairwise_comparisons])
    # rdc_vals = []
    # for i, j in pairwise_comparisons:
    #     rdc_vals.append(rdc_cca((i, j)))

    pairwise_comparisons = itertools.combinations(np.arange(n_features), 2)
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        print(rdc)
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    #
    # setting diagonal to 1
    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1
    print(rdc_adjacency_matrix)

    return rdc_adjacency_matrix


# def rdc_cca(i, j):
def rdc_cca(indexes):
    i, j, cca = indexes
    # TODO 3) Zeile unnötig? cca wird übergeben?
    cca = CCA(n_components=1)
    X_cca, Y_cca = cca.fit_transform(GLOBAL_RDC_FEATURES[i], GLOBAL_RDC_FEATURES[j])

    # rdc = 1
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    print('ij', i, j)
    return rdc


def rdc_cca2(i, j):
    cca = CCA(n_components=1)
    # TODO 5) die komische Zeile:
    #  input 2 Matrizen,
    #  output 2 vectoren 20 x 1 mit sehr kleinen Werten
    X_cca, Y_cca = cca.fit_transform(GLOBAL_RDC_FEATURES[i], GLOBAL_RDC_FEATURES[j])
    # TODO 6) die andere komische Zeile
    #  input die 2 Vektoren von oben
    #  output der rdc value
    rdc = np.corrcoef(X_cca.T, Y_cca.T)[0, 1]
    return rdc


def rdc_test2(rdc_features, cols, n_jobs=7):
    n_features = len(cols)

    # build adjacency matrix
    rdc_adjacency_matrix = np.zeros((n_features, n_features))

    GLOBAL_RDC_FEATURES.clear()
    GLOBAL_RDC_FEATURES.extend(rdc_features)

    # TODO 2) Das printen von pairwise_comparison verfälscht das Ergebnis
    # for i, j in pairwise_comparisons:
    #    print("i: ", i, "j: ", j, "\n")
    pairwise_comparisons = itertools.combinations(np.arange(n_features), 2)

    #cca = CCA(n_components=1)
    #from joblib import Parallel, delayed
    #rdc_vals = Parallel(n_jobs=n_jobs)(delayed(rdc_cca)((i, j, cca)) for i, j in pairwise_comparisons)

    # TODO 4) alternativer code, ist aber vermutlich unperformanter ?
    rdc_vals = []
    for i, j in pairwise_comparisons:
        rdc = rdc_cca2(i,j)
        print("i:", i, "j:", j, "rdc:", rdc, "\n")
        rdc_vals.append(rdc)

    pairwise_comparisons = itertools.combinations(np.arange(n_features), 2)
    for (i, j), rdc in zip(pairwise_comparisons, rdc_vals):
        rdc_adjacency_matrix[i, j] = rdc
        rdc_adjacency_matrix[j, i] = rdc

    rdc_adjacency_matrix[np.diag_indices_from(rdc_adjacency_matrix)] = 1
    return rdc_adjacency_matrix


if __name__ == '__main__':
    k = 20
    s = 1. / 6.
    non_linearity = np.sin
    rand_gen = np.random.RandomState(RAND_STATE)
    return_matrix = False

    # DATA SLICE
    filename = "res/t3/t3.csv";
    gaussian_path = "res/t3/gaussian_0";

    #filename = "res/t4/t4.csv";
    #gaussian_path = "res/t4/gaussian_0";

    data, _ = readCsvAsNumpyArrayAndHeader(filename)
    rows = np.arange(data.shape[0])
    cols = np.arange(data.shape[1])

    # TODO 1) hier wird ne 1 Spalte hintendrangehängt also rows x k+1 große Matrizen
    rdc_features = rdc_transformer(data, rows, cols, gaussian_path, k, s, non_linearity, rand_gen, False)
    rdc_adjacency_matrix = rdc_test2(rdc_features, cols)
    print(rdc_adjacency_matrix)
