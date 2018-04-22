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
        if instance[target_attr].decode() != classifier.classify_normalization(instance):
            errors += 1
        else:
            hits += 1
    return hits / len(validation_set)


def scale(data, attributes, use_standarization):
    # 'data' es es el conjunto de entrenamiento
    # 'attributes' un array con los nombres de los atributos
    # si 'use_standarization' = true, normaliza usando media y varianza y retorna (data, media, varianza)
    # si no, usa min-max y retorna (data, min, max)
    numeric_attributes_values = {}
    # Se recorre el data para extraer los valores de los distintos atributos numéricos
    for instance in data:
        for attribute in attributes:
            #  Si se encuentra un valor numérico se agrega al diccionario
            if isinstance(instance[attribute], np.float64):
                if attribute in numeric_attributes_values:
                    numeric_attributes_values[attribute].append(instance[attribute])
                else:
                    numeric_attributes_values[attribute] = [instance[attribute]]
    print()
    print('first',data)
    print()
    for att, values in numeric_attributes_values.items():
        values_np = np.asarray(values)
        if use_standarization:
            scaled = (values_np - values_np.mean()) / values_np.std()
            i = -1
            for instance in data:
                i += 1
                instance[att] = scaled[i]
        else:
            scaled = (values_np - values_np.min()) / (values_np.max() - values_np.min())
            i = -1
            for instance in data:
                i += 1
                instance[att] = scaled[i]
    print()
    print('second',data)


