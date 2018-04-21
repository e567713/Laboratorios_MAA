import math
import numpy as np
import utils
import random


def get_k_nearest(instance, k, data, target_attr):
  # Devuelve una tupla t = (distances, instances) ordenada de menor a mayor por distances de 
  # largo k siendo 'instances' las 'k' instancias más cercanas de 'data' y 'distances' 
  # son las distancias de las 'instances'.
  # 'target_attr' es el nombre del atributo objetivo

  distances_array = []
  dis_inst = []
  i = -1
  for example in data:
    i += 1
    distances_array.append((i, distance(instance, example, target_attr)))
  distances_array.sort(key = lambda tup: tup[1])
  # print()
  # print('(indice, distancia) ordenado de menor a mayor')
  # for dist in distances_array: print(dist)
  # print()
  for element in distances_array[:k]:
    dis_inst.append((element[1], data[element[0]]))
  return dis_inst


def distance(instance1, instance2, target_attr):
  # Calcula la distancia euclidiana entre las 2 instancias
  # 'target_attr' es el nombre del atributo objetivo

  # Descomentar para el training set hecho por nosotros
  # instance1_values = list(instance1.values())
  # instance2_values = list(instance2.values())

  att_length = len(instance1) - 1 if target_attr in instance1.dtype.names else len(instance1)
  
  sumatory = 0
  for i in range (att_length):
      if isinstance(instance1[i], np.float64):
        sumatory += pow(abs(instance2[i] - instance1[i]), 2)
      elif instance1[i] != instance2[i]:
        sumatory += 1
  return math.sqrt(sumatory)


def classify(instance, data, k, target_attr, weight):
  # Clasifica la instancia 'instance' para el conjunto de entrenamiento 'data', utilizando los 'k' casos más cercanos 
  # 'target_attr' es el nombre del atributo objetivo
  # si 'weight' es true  se usa pesos para los 'k' más cercanos
  nearest = get_k_nearest(instance, k, data, target_attr)

  print()
  print(str(k) + '-nearest:')
  for instance in nearest: print(instance)
  print()

  return find_most_common_function_value(nearest, target_attr, weight)


def find_most_common_function_value(data, target_attr, use_weight):
  # data es una tupla t (distances, instance), la tupla debe estar ordenada de menor a mayor por distancias
  # si 'use_weight' es false devuelve el valor más común del atributo 'target_attr' en el conjunto 'data'
  # si 'use_weight' es true se calcula en base a pesos
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
  print(count_dict)
  for key, value in count_dict.items():
    if value > max_qty:
      max_qty = value
      max_values = []
      max_values.append(key)
    elif value == max_qty:
        max_values.append(key)
  max_v = random.choice(max_values)
  return max_v.decode() if not isinstance(max_v, str) else max_v


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

attributes = ['A1_Score',
              'A2_Score',
              'A3_Score',
              'A4_Score',
              'A5_Score',
              'A6_Score',
              'A7_Score',
              'A8_Score',
              'A9_Score',
              'A10_Score',
              'age',
              'gender',
              'ethnicity',
              'jundice',
              'austim',
              'contry_of_res',
              'used_app_before',
              'result',
              'age_desc',
              'relation']



# print ('La nueva instancia sería:', classify(instance, S, 2, 'Salva', False))


examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos


# data_set = utils.process_missing_values(data_set, attributes, False)
# utils.scale(data_set, attributes, True)


x = [1,2,3]
values_np = np.asarray(x)
standarized = (values_np - values_np.mean()) / values_np.std()
mean = values_np.mean()
std = values_np.std()
print ((3 - mean) / std)
print (standarized)

# # Primeros 9 ejemplos
# first_nine_examples = data_set[:9]
# # Decimo ejemplo
# tenth_example = data_set[9]
# print()
# print('Conj de entrenamiento:\n', first_nine_examples)
# print()
# print('instancia a clasificar', tenth_example)
# print()
# print()
# print ('El resultado de la nueva instancia sería:', classify(tenth_example, first_nine_examples, 5, 'Class/ASD', True))
# print()




#######################################################################################
###########################           Parte 1          ################################
#######################################################################################