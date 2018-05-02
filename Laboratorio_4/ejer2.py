import numpy as np
import scipy

def PCA(data, attributes):
  # Está hecho para 2 dimensiones
  # Se recorre el data para extraer como array los valores de los distintos atributos numéricos
  numeric_attributes_values = {}
  n = len(data)
  for instance in data:
    for attribute in attributes:
      #  Si se encuentra un valor numérico se agrega al diccionario
      if isinstance(instance[attribute], np.float64) or isinstance(instance[attribute], float):
        if attribute in numeric_attributes_values:
          numeric_attributes_values[attribute].append(instance[attribute])
        else:
          numeric_attributes_values[attribute] = [instance[attribute]]

  if (len(numeric_attributes_values.keys()) != 2):
    return print('Este método está hecho para 2 dimensiones')
  
  for att, values in numeric_attributes_values.items():
    values_np = np.asarray(values)
    mean = values_np.mean()
    i = -1
    for val in values:
      i += 1
      # data[i][att] -= mean
      values[i] = val - mean

  cov_x_x = 0
  cov_x_y = 0
  cov_y_y = 0
  for i in range(n):
    cov_x_x += numeric_attributes_values[attributes[0]][i] * numeric_attributes_values[attributes[0]][i]
    cov_x_y += numeric_attributes_values[attributes[0]][i] * numeric_attributes_values[attributes[1]][i]
    cov_y_y += numeric_attributes_values[attributes[1]][i] * numeric_attributes_values[attributes[1]][i]
  # TODO ver si se divide por n - 1 o por n
  cov_x_x /= (n - 1)
  cov_x_y /= (n - 1)
  cov_y_y /= (n - 1)

  # matriz de covarianza 
  #     FILA 0        FILA 1
  # [[col0, col1], [col0, col1]]
  covariance_matrix = np.array([[cov_x_x, cov_x_y], [cov_x_y, cov_y_y]])
  u, s, v = np.linalg.svd(covariance_matrix)
  print(u)
  print(s)
  print(v)

  


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
PCA(a, atts)
 