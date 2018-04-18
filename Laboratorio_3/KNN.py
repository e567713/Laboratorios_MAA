import math
import numpy as np
import utils


def get_k_nearest(instance, k, data):
  # Devuelve una tupla t = (i,d) donde i es el indice de la instancia en data y d la distancia 
  distances_array = []
  instances = []
  i = -1
  for example in data:
    i += 1
    distances_array.append((i, distance(instance, example)))
  distances_array.sort(key = lambda tup: tup[1])
  for element in distances_array[:k]:
    instances.append(data[element[0]])
  return instances


def distance(instance1, instance2):
  # Calcula la distancia euclidiana entre las 2 instancias
  instance1_values = list(instance1.values())
  instance2_values = list(instance2.values())
  att_length = len(instance1_values) if len(instance1_values) < len(instance2_values) else len(instance2_values)
  sumatory = 0
  for i in range (att_length):
      if isinstance(instance1_values[i], np.float64) or isinstance(instance1_values[i], int):
        sumatory += pow(abs(instance2_values[i] - instance1_values[i]), 2)
      elif instance1_values[i] != instance2_values[i]:
        sumatory += 1

  return math.sqrt(sumatory)


def classify(instance, data, k, target_attr):
  # Clasifica la instancia instance para el conjunto de entrenamiento data
  nearest = get_k_nearest(instance, k, data)
  return utils.find_most_common_function_value(nearest, target_attr)


def normalize(instance):
  # TODO
  return


S = [
  {'Dedicacion': 0, 'Dificultad': 'Alta', 'Horario': 'Nocturno',
      'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
  {'Dedicacion': 20, 'Dificultad': 'Media', 'Horario': 'Matutino',
      'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
  {'Dedicacion': 10, 'Dificultad': 'Alta', 'Horario': 'Nocturno',
      'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
  {'Dedicacion': 2, 'Dificultad': 'Alta', 'Horario': 'Matutino',
      'Humedad': 'Alta', 'Humor Docente': 'Bueno', 'Salva': 'No'},
]

instance = {'Dedicacion': 0, 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Bueno'}


print ('La nueva instancia sería:', classify(instance, S, 2, 'Salva'))