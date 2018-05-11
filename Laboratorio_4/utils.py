
from scipy.io import arff
import numpy as np
from collections import Counter
import copy

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


def process_missing_values(data, attributes, use_most_common):
    # Procesa el conjunto de datos para atacar el problema de valores incompletos.

    # Guardará los valores más comunes para los atributos que necesiten calcularse.
    most_common = {}
    i = len(data)

    # Se itera de atrás para adelante para poder borrar por indice si not use_most_common
    for instance in reversed(data):
        i -= 1
        for attribute in attributes:

            # Si se encuentra un valor faltante.
            if ((isinstance(instance[attribute], np.float64) and np.isnan(instance[attribute])) or (isinstance(instance[attribute], np.bytes_) and instance[attribute] == b'?')):
                # Si use_most_common, se cambia el valor faltante por el más común del atributo,
                # si no, se descarta el ejemplo
                if use_most_common:
                    # Si no se ha calculado el valor más común, se lo calcula y almacena.
                    if not attribute in most_common:
                        most_common[attribute]=found_most_common_attribute_value(
                            data, attribute)

                    # Se cambia el valor faltante por el más común.
                    instance[attribute] = most_common[attribute]
                else:
                    # Se descarta el ejemplo
                    data = np.delete(data,i)
                    break
    return data


def found_most_common_attribute_value(data, attribute):
    values = [instance[attribute] for instance in data if not (isinstance(instance[attribute], np.float64) and np.isnan(instance[attribute])) or (isinstance(instance[attribute], np.bytes_) and instance[attribute] == b'?')]
    data = Counter(values)
    return max(values, key=data.get)



def split_20_80(d):
    # Divide el conjunto de datos en dos subconjuntos, uno con el 80% de los datos
    # y el otro con el restante 20%.

    # Se copia el conjunto de datos original para no alterarlo.
    data = copy.deepcopy(d)

    # Se ordenan aleatoriamente los ejemplos del conjunto para simular la
    # elección al azar de elementos para formar los subconjuntos.
    np.random.shuffle(data)

    # Se obtiene la cantidad de elementos que suponen el 20% de los datos.
    limit = len(data) // 5

    # Se crean los subconjuntos
    subset_20 = data[:limit]
    subset_80 = data[limit:]

    return (subset_20, subset_80)

def process_numeric_values(data, numeric_attributes, target_attr):

    for numeric_attr in numeric_attributes:
        # Se calcula el mejor threshold para el atributo numérico numeric_attr
        # midiendo según la gananacia de información.

        thresholds = []

        # Se ordenan los ejemplos en orden ascendente según los valores de numeric_attr.
        sorted_data = sorted(data, key=lambda x: x[numeric_attr])

        # Se recorre el conjunto data comparando de a 2 elementos para encontrar posibles
        # thresholds.
        for i in range(0, len(sorted_data) - 1):
            instance_1 = sorted_data[i]
            instance_2 = sorted_data[i + 1]

            # En caso de encontrar un posible candidato se almacena
            if instance_1[target_attr] != instance_2[target_attr] and instance_1[numeric_attr] != instance_2[numeric_attr]:
                thresholds.append(
                    (instance_1[numeric_attr] + instance_2[numeric_attr]) / 2)

        # Se recorre la lista de posibles thresholds.
        for threshold in thresholds:

            # Se dividen los valores de numeric_attr según el threshold dado.
            splitted_data = set_numeric_attribute_values(
                copy.deepcopy(data), numeric_attr, threshold)

            # Se busca el threshold que maximiza la ganancia de información.
            maximum_thresholds_tied = []
            max_ig = -1
            ig = information_gain(splitted_data, numeric_attr, target_attr, use_missing_values_first_method)
            if ig > max_ig:
                max_ig = ig
                maximum_thresholds_tied = []
                maximum_thresholds_tied.append(splitted_data)
            elif ig == max_ig:
                maximum_thresholds_tied.append(splitted_data)

        best_splitted_data = random.choice(maximum_thresholds_tied)

        # Se setean los valores del conjunto de datos según los resultados obtenidos
        # por el mejor threshold.
        for i in range(len(data)):
            data[i][numeric_attr] = best_splitted_data[i][numeric_attr]

    return data
        
def set_numeric_attribute_values(data, numeric_attr, threshold):
    # Divide los valores del atributo numérico numeric_attr según
    # el valor del threshold pasado por parámetro.

    new_key = numeric_attr + '>' + str(threshold)

    for instance in data:
        if instance[numeric_attr] > threshold:
            instance[numeric_attr] = 1
        else:
            instance[numeric_attr] = 0

    return data
