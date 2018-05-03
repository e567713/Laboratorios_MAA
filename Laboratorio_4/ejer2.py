import numpy as np
import scipy
import copy

def PCA(data, attributes, numeric_atts):
  # Se recorre el data para extraer como array los valores de los distintos atributos numéricos
  data_copy = copy.deepcopy(data)
  numeric_attributes_values = {}
  data_len = len(data)
  atts_len = len(attributes)
  for instance in data:
    for attribute in numeric_atts:
      #  Si se encuentra un valor numérico se agrega al diccionario
      if isinstance(instance[attribute], np.float64) or isinstance(instance[attribute], float) or isinstance(instance[attribute], int):
        if attribute in numeric_attributes_values:
          numeric_attributes_values[attribute].append(instance[attribute])
        else:
          numeric_attributes_values[attribute] = [instance[attribute]]
  
  for att, values in numeric_attributes_values.items():
    values_np = np.asarray(values)
    mean = values_np.mean()
    i = -1
    for val in values:
      i += 1
      data_copy[i][att] -= mean
      # values[i] = val - mean

  # matriz de covarianza 
  #     FILA 0        FILA 1
  # [[col0, col1], [col0, col1]]
  matrix = []
  for k in range(atts_len):
    row = []
    for i in range(data_len):
      for j in range(atts_len):
        if len(row) < j+1:
          row.append(data_copy[i][attributes[k]] * data_copy[i][attributes[j]])
        else:
          row[j] += data_copy[i][attributes[k]] * data_copy[i][attributes[j]]

    for index in range(len(row)):
      row[index] /= (data_len - 1)
    matrix.append(row)

  covariance_matrix = np.array(matrix)
  print('covariance matrix')
  print(covariance_matrix)

  eigen_vegtors, eigen_values, v = np.linalg.svd(covariance_matrix)
  print(eigen_vegtors)
  print(eigen_values)

  


a = [{'x': 2.5, 'y': 2.4},
     {'x': 0.5, 'y': 0.7},
     {'x': 2.2, 'y': 2.9},
     {'x': 1.9, 'y': 2.2},
     {'x': 3.1, 'y': 3.0},
     {'x': 2.3, 'y': 2.7},
     {'x': 2.0, 'y': 1.6},
     {'x': 1.0, 'y': 1.1},
     {'x': 1.5, 'y': 1.6},
     {'x': 1.1, 'y': 0.9}]

atts = ['x', 'y']

# a = [{'x': 2.5, 'y': 2.4, 'z': 2.1},
#      {'x': 0.5, 'y': 0.7, 'z': 0.9},
#      {'x': 2.2, 'y': 2.9, 'z': 2.5},
#      {'x': 1.9, 'y': 2.2, 'z': 1.7},
#      {'x': 3.1, 'y': 3.0, 'z': 3.5},
#      {'x': 2.3, 'y': 2.7, 'z': 2.2},
#      {'x': 2.0, 'y': 1.6, 'z': 1.9},
#      {'x': 1.0, 'y': 1.1, 'z': 1.0},
#      {'x': 1.5, 'y': 1.6, 'z': 0.8},
#      {'x': 1.1, 'y': 0.9, 'z': 0.9}]

# atts = ['x', 'y', 'z']

PCA(a, atts, atts)
 