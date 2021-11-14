import numpy as np
import os

def readCsvAsNumpyArrayAndHeader(filename:str):
    f = open(filename)

    header_line = f.readline()
    header_line = header_line.strip(',\n')
    column_names = header_line.split(',')

    mat = []
    for line in f:
        tokens = line.rstrip().split(',')
        mat.append([float(n) for n in tokens])

    f.close()
    return np.array(mat), column_names

def readCsvAsNumpyArray(filename:str):
    f = open(filename)

    mat = []
    for line in f:
        tokens = line.rstrip().split(',')
        mat.append([float(n) for n in tokens])

    f.close()
    return np.array(mat)

if __name__ == '__main__':
    path = os.getcwd()
    print(path)
    print(type(path))

    m, columNames = readCsvAsNumpyArray("res/t4/t4.csv")
    print(columNames)
    print(m)

