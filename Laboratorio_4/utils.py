from scipy.io import arff
import numpy as np

def read_file(path):
    return arff.loadarff(path)


def decode_data (data):
  # Devuelve un arreglo de arreglos donde cada elemento es una fila
  result = []
  for i in range(len(data)):
    result.append([])
    for j in range(len(data[i])):
      if(isinstance(data[i][j], np.bytes_)):
        result[i].append(data[i][j].decode())
      else:
        result[i].append(data[i][j])
  return result