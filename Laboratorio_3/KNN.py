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


def validate(data, validation_set, target_attr, attributes, k, weight):
  errors = 0
  for instance in validation_set:
    if classify(instance, data, k, target_attr, weight, attributes) != instance[target_attr]:
      errors += 1
  return errors




if __name__ == "__main__":
  #######################################################################################
  ###########################            MAIN            ################################
  #######################################################################################


  # #######################################################################################
  # ###########################     Validación cruzada     ################################
  # #######################################################################################

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
                'age_desc',
                'relation']

  examples = utils.read_file('Autism-Adult-Data.arff')
  data_set = examples[0]  # Datos
  metadata = examples[1]  # Metadatos

  data = copy.deepcopy(data_set)
  # Se desordena el conjunto de datos y se parte 20-80
  splitted_data = utils.split_20_80(data)

  print()
  print('Hay 24 variantes del algoritmo KNN')
  print('Cada ejecución lleva su tiempo...')
  print()


  #######################################################################################
  ###########################            K = 1           ################################
  #######################################################################################
  

  print()
  print('1) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, usa pesos y normaliza utilizando standarization')
  print()

  # Me quedo con el 80% del conjunto
  data1 = copy.deepcopy(splitted_data[1])
  data1 = utils.process_missing_values(data1, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result1 = utils.cross_validation(data1, attributes, 'Class/ASD', 10, True, 1, True, True, True)



  print()
  print('2) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data2 = copy.deepcopy(splitted_data[1])
  data2 = utils.process_missing_values(data2, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result2 = utils.cross_validation(data2, attributes, 'Class/ASD', 10, True, 1, True, True, False)



  print()
  print('3) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data3 = copy.deepcopy(splitted_data[1])
  data3 = utils.process_missing_values(data3, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result3 = utils.cross_validation(data3, attributes, 'Class/ASD', 10, True, 1, False, True, True)



  print()
  print('4) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data4 = copy.deepcopy(splitted_data[1])
  data4 = utils.process_missing_values(data4, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result4 = utils.cross_validation(data4, attributes, 'Class/ASD', 10, True, 1, False, True, False)



  print()
  print('5) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data5 = copy.deepcopy(splitted_data[1])
  data5 = utils.process_missing_values(data5, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result5 = utils.cross_validation(data5, attributes, 'Class/ASD', 10, True, 1, False, True, True)



  print()
  print('6) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data6 = copy.deepcopy(splitted_data[1])
  data6 = utils.process_missing_values(data6, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result6 = utils.cross_validation(data6, attributes, 'Class/ASD', 10, True, 1, True, True, False)



  print()
  print('7) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data7 = copy.deepcopy(splitted_data[1])
  data7 = utils.process_missing_values(data7, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result7 = utils.cross_validation(data7, attributes, 'Class/ASD', 10, True, 1, False, True, True)
  


  print()
  print('8) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data8 = copy.deepcopy(splitted_data[1])
  data8 = utils.process_missing_values(data8, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result8 = utils.cross_validation(data8, attributes, 'Class/ASD', 10, True, 1, False, True, False)
  

  #######################################################################################
  ###########################            K = 3           ################################
  #######################################################################################


  print()
  print('9) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, usa pesos y normaliza utilizando standarization')
  print()

  # Me quedo con el 80% del conjunto
  data9 = copy.deepcopy(splitted_data[1])
  data9 = utils.process_missing_values(data9, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result9 = utils.cross_validation(data9, attributes, 'Class/ASD', 10, True, 3, True, True, True)



  print()
  print('10) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data10 = copy.deepcopy(splitted_data[1])
  data10 = utils.process_missing_values(data10, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result10 = utils.cross_validation(data10, attributes, 'Class/ASD', 10, True, 3, True, True, False)



  print()
  print('11) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data11 = copy.deepcopy(splitted_data[1])
  data11 = utils.process_missing_values(data11, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result11 = utils.cross_validation(data11, attributes, 'Class/ASD', 10, True, 3, False, True, True)



  print()
  print('12) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data12 = copy.deepcopy(splitted_data[1])
  data12 = utils.process_missing_values(data12, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result12 = utils.cross_validation(data12, attributes, 'Class/ASD', 10, True, 3, False, True, False)



  print()
  print('13) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data13 = copy.deepcopy(splitted_data[1])
  data13 = utils.process_missing_values(data13, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result13 = utils.cross_validation(data13, attributes, 'Class/ASD', 10, True, 3, False, True, True)



  print()
  print('14) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data14 = copy.deepcopy(splitted_data[1])
  data14 = utils.process_missing_values(data14, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result14 = utils.cross_validation(data14, attributes, 'Class/ASD', 10, True, 3, True, True, False)



  print()
  print('15) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data15 = copy.deepcopy(splitted_data[1])
  data15 = utils.process_missing_values(data15, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result15 = utils.cross_validation(data15, attributes, 'Class/ASD', 10, True, 3, False, True, True)
  


  print()
  print('16) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data16 = copy.deepcopy(splitted_data[1])
  data16 = utils.process_missing_values(data16, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result16 = utils.cross_validation(data16, attributes, 'Class/ASD', 10, True, 3, False, True, False)


  #######################################################################################
  ###########################            K = 7           ################################
  #######################################################################################


  print()
  print('17) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, usa pesos y normaliza utilizando standarization')
  print()

  # Me quedo con el 80% del conjunto
  data17 = copy.deepcopy(splitted_data[1])
  data17 = utils.process_missing_values(data17, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result17 = utils.cross_validation(data17, attributes, 'Class/ASD', 10, True, 7, True, True, True)



  print()
  print('18) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data18 = copy.deepcopy(splitted_data[1])
  data18 = utils.process_missing_values(data18, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result18 = utils.cross_validation(data18, attributes, 'Class/ASD', 10, True, 7, True, True, False)



  print()
  print('19) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data19 = copy.deepcopy(splitted_data[1])
  data19 = utils.process_missing_values(data19, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result19 = utils.cross_validation(data19, attributes, 'Class/ASD', 10, True, 7, False, True, True)



  print()
  print('20) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data20 = copy.deepcopy(splitted_data[1])
  data20 = utils.process_missing_values(data20, attributes, True)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result20 = utils.cross_validation(data20, attributes, 'Class/ASD', 10, True, 7, False, True, False)



  print()
  print('21) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data21 = copy.deepcopy(splitted_data[1])
  data21 = utils.process_missing_values(data21, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result21 = utils.cross_validation(data21, attributes, 'Class/ASD', 10, True, 7, False, True, True)



  print()
  print('22) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data22 = copy.deepcopy(splitted_data[1])
  data22 = utils.process_missing_values(data22, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result22 = utils.cross_validation(data22, attributes, 'Class/ASD', 10, True, 7, True, True, False)



  print()
  print('23) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando standarization')
  print()
  
  # Me quedo con el 80% del conjunto
  data23 = copy.deepcopy(splitted_data[1])
  data23 = utils.process_missing_values(data23, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result23 = utils.cross_validation(data23, attributes, 'Class/ASD', 10, True, 7, False, True, True)
  


  print()
  print('24) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando min-max')
  print()
  
  # Me quedo con el 80% del conjunto
  data24 = copy.deepcopy(splitted_data[1])
  data24 = utils.process_missing_values(data24, attributes, False)
                # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
  result24 = utils.cross_validation(data24, attributes, 'Class/ASD', 10, True, 7, False, True, False)



  print("-------------------------------------------------------------------------------------")
  print("KNN")
  print()
  print("Tasa de aciertos sobre un conjunto de validación:")
  print ("result 1: ")
  print("\t",1-result1)
  print ("result 2: ")
  print("\t",1-result2)
  print ("result 3: ")
  print("\t",1-result3)
  print ("result 4: ")
  print("\t",1-result4)
  print ("result 5: ")
  print("\t",1-result5)
  print ("result 6: ")
  print("\t",1-result6)
  print ("result 7: ")
  print("\t",1-result7)
  print ("result 8: ")
  print("\t",1-result8)
  print ("result 9: ")
  print("\t",1-result9)
  print ("result 10: ")
  print("\t",1-result10)
  print ("result 11: ")
  print("\t",1-result11)
  print ("result 12: ")
  print("\t",1-result12)
  print ("result 13: ")
  print("\t",1-result13)
  print ("result 14: ")
  print("\t",1-result14)
  print ("result 15: ")
  print("\t",1-result15)
  print ("result 16: ")
  print("\t",1-result16)
  print ("result 17: ")
  print("\t",1-result17)
  print ("result 18: ")
  print("\t",1-result18)
  print ("result 19: ")
  print("\t",1-result19)
  print ("result 20: ")
  print("\t",1-result20)
  print ("result 21: ")
  print("\t",1-result21)
  print ("result 22: ")
  print("\t",1-result22)
  print ("result 23: ")
  print("\t",1-result23)
  print ("result 24: ")
  print("\t",1-result24)
  print("-------------------------------------------------------------------------------------")