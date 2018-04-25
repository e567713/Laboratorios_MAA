import math
import numpy as np
import utils
import random
import copy
import threading


def get_k_nearest(instance, k, data, attributes, target_attr):
  # Devuelve una tupla t = (distances, instances) ordenada de menor a mayor por distances de 
  # largo k siendo 'instances' las 'k' instancias más cercanas de 'data' y 'distances' 
  # son las distancias de las 'instances'.
  # 'target_attr' es el nombre del atributo objetivo

  distances_array = []
  dis_inst = []
  i = -1
  for example in data:
    i += 1
    distances_array.append((i, distance(instance, example, attributes, target_attr)))
  distances_array.sort(key = lambda tup: tup[1])
  # print()
  # print('(indice, distancia) ordenado de menor a mayor')
  # for dist in distances_array: print(dist)
  # print()
  for element in distances_array[:k]:
    dis_inst.append((element[1], data[element[0]]))
  return dis_inst


def distance(instance1, instance2, attributes, target_attr):
  # Calcula la distancia euclidiana entre las 2 instancias
  # Si el atributo no es numerico, la distancia es 1 si tiene diferentes valores
  # 'target_attr' es el nombre del atributo objetivo
  sumatory = 0
  for attribute in attributes:
      if isinstance(instance1[attribute], np.float64):
        sumatory += pow(abs(instance2[attribute] - instance1[attribute]), 2)
      elif instance1[attribute] != instance2[attribute]:
        sumatory += 1
  return math.sqrt(sumatory)


def classify(instance, data, k, target_attr, weight, attributes):
  # Clasifica la instancia 'instance' para el conjunto de entrenamiento 'data', utilizando los 'k' casos más cercanos 
  # 'target_attr' es el nombre del atributo objetivo
  # si 'weight' es true se usa pesos para los 'k' más cercanos
  nearest = get_k_nearest(instance, k, data, attributes, target_attr)

  # print()
  # print(str(k) + '-nearest:')
  # for instance in nearest: print(instance)
  # print()

  return find_most_common_function_value(nearest, target_attr, weight)


def find_most_common_function_value(data, target_attr, use_weight):
  # data es una tupla t (distances, instance), la tupla debe estar ordenada de menor a mayor por distancias
  # si 'use_weight' = false devuelve el valor más común del atributo 'target_attr' en el conjunto 'data'
  # si 'use_weight' = true se calcula en base a pesos
  distances = [x[0] for x in data]
  instances = [x[1] for x in data]
  count_dict = {}
  max_values = []
  max_qty = 0
  i = -1
  exist_same_example = False
  for instance in instances:
    if not use_weight:
      if (instance[target_attr] in count_dict):
        count_dict[instance[target_attr]] += 1
      else:
        count_dict[instance[target_attr]] = 1
    else:
      i += 1
      weight = None
      if distances[i] == 0:
        # Si es igual a un ejemplo del conjunto de entrenamiento, agarro todos los ejemplos
        # que sean iguales y me quedo con el 'target_attribute' más comun
        exist_same_example = True
      elif distances[i] != 0 and exist_same_example:
        break
      elif distances[i] != 0:
        weight = 1 / pow(distances[i], 2)

      if (instance[target_attr] in count_dict):
        if weight:
          count_dict[instance[target_attr]] += weight * 1
        else:
          count_dict[instance[target_attr]] += 1
      else:
        if weight:
          count_dict[instance[target_attr]] = weight * 1
        else:  
          count_dict[instance[target_attr]] = 1
  
  for key, value in count_dict.items():
    if value > max_qty:
      max_qty = value
      max_values = []
      max_values.append(key)
    elif value == max_qty:
        max_values.append(key)
  max_v = random.choice(max_values)
  return max_v


def holdout_validation(data, validation_set, target_attr, attributes, k, weight, normalize, use_standarization):
  # retorna (len(validation_set), cantidad de errores, promedio de errores)
  errors = 0
  if normalize:
    scale_values = utils.scale(data, attributes, use_standarization)
    data = scale_values[0]
    scalation_parameters = scale_values[1]
  for instance in validation_set:
    instance_copy = copy.deepcopy(instance)
    if normalize:
      instance_copy = utils.scale_instance(instance_copy, scalation_parameters, use_standarization)
    if classify(instance_copy, data, k, target_attr, weight, attributes) != instance[target_attr]:
      errors += 1
  return (len(validation_set), errors, errors/len(validation_set))
