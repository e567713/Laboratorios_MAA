from scipy.io import arff

def read_file(path):
    return arff.loadarff(path)