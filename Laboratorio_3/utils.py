# coding=utf-8
from scipy.io import arff
import copy
import random
from collections import Counter
import numpy as np


def read_file(path):
    return arff.loadarff(path)


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


def process_missing_values(data, attributes):
    # Procesa el conjunto de datos para atacar el problema de valores incompletos.

    # Guardará los valores más comunes para los atributos que necesiten calcularse.
    most_common = {}

    for instance in data:
        for attribute in attributes:

            # Si se encuentra un valor faltante.
            if ((isinstance(instance[attribute], np.float64) and np.isnan(instance[attribute])) or (isinstance(instance[attribute], np.bytes_) and instance[attribute] == b'?')):

                # Si no se ha calculado el valor más común, se lo calcula y almacena.
                if not attribute in most_common:
                    most_common[attribute]=found_most_common_attribute_value(
                        data, attribute)

                # Se cambia el valor faltante por el más común.
                instance[attribute] = most_common[attribute]
    return data


def found_most_common_attribute_value(data, attribute):
    values = [instance[attribute] for instance in data if not (isinstance(instance[attribute], np.float64) and np.isnan(instance[attribute])) or (isinstance(instance[attribute], np.bytes_) and instance[attribute] == b'?')]
    data = Counter(values)
    return max(values, key=data.get)


def process_numeric_values_discretize(data, attributes):
    # Se discretizan las edades en decadas para que no queden los atributos
    # dispersos y con poca probabilidad.
    # OBS: Evita la columna 'result'.
    for instance in data:
        instance['age'] = instance['age'] // 10
    return data

def process_numeric_values_normalize(data, attributes):
    # Se discretizan las edades en decadas para que no queden los atributos
    # dispersos y con poca probabilidad.
    # OBS: Evita la columna 'result'.

    #Asumimos que tienen una distribucion gaussiana
    for instance in data:
        instance['age'] = instance['age'] // 10
    return data


def validate(validation_set , classifier, target_attr):
    # Realiza la validación del clasificador.
    
    if len(validation_set) == 0:
        return 1

    errors = 0
    hits = 0

    for instance in validation_set:
        if instance[target_attr].decode() != classifier.classify(instance):
            errors += 1
        else:
            hits += 1
    return hits / len(validation_set)


