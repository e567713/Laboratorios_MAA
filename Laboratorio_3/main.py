import utils
from naive_bayes import NaiveBayes
import copy
import numpy as np


# Data set del teórico
S = [
    {'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
    {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Malo', 'Salva': 'Yes'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Bueno', 'Salva': 'No'},
]

# Atributos a tener en cuenta
target_attr = 'Class/ASD'
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


#######################################################################################
################           Carga y procesamiento de datos          ####################
#######################################################################################
 
# Leemos data set del laboratorio
examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos
# Se procesan los valores faltantes
# utils.process_missing_values(data_set,attributes, True)

# Se procesan los valores numéricos
# TODO
# utils.process_numeric_values_discretize(data_set,attributes)

# Separamos el data set en dos subconjuntos
# splitted_data = utils.split_20_80(data_set)
#
# validation_set = splitted_data[0]
# training_set = splitted_data[1]
#


#######################################################################################
###########################           Parte 1          ################################
#######################################################################################

# Se copia el conjunto de datos original para no alterarlo.
data = copy.deepcopy(data_set)
# Se ordenan aleatoriamente los ejemplos del conjunto para simular la
# elección al azar de elementos para formar los subconjuntos.
np.random.shuffle(data)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# sin discretizar los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data1 = copy.deepcopy(data)
data1 = utils.process_missing_values(data1,attributes, True)
result1 = utils.cross_validation(data1, attributes, target_attr, 4, False, 0, False, False)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# sin discretizar los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data2 = copy.deepcopy(data)
data2 = utils.process_missing_values(data2,attributes, False)
result2 = utils.cross_validation(data2, attributes, target_attr, 4, False, 0, False, False)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# discretizando los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data3 = copy.deepcopy(data)
data3 = utils.process_missing_values(data3,attributes, True)
data3 = utils.process_numeric_values_discretize(data3,attributes)
result3 = utils.cross_validation(data3, attributes, target_attr, 4, False, 0, False, False)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# discretizando los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data4 = copy.deepcopy(data)
data4 = utils.process_missing_values(data4,attributes, False)
data4 = utils.process_numeric_values_discretize(data4,attributes)
result4 = utils.cross_validation(data4, attributes, target_attr, 4, False, 0, False, False)



# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# sin discretizar los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data5 = copy.deepcopy(data)
data5 = utils.process_missing_values(data5,attributes, True)
result5 = utils.cross_validation(data5, attributes, target_attr, 4, False, 0, False, True)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# sin discretizar los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data6 = copy.deepcopy(data)
data6 = utils.process_missing_values(data6,attributes, False)
result6 = utils.cross_validation(data6, attributes, target_attr, 4, False, 0, False, True)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# discretizando los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data7 = copy.deepcopy(data)
data7 = utils.process_missing_values(data7,attributes, True)
data7 = utils.process_numeric_values_discretize(data7,attributes)
result7 = utils.cross_validation(data7, attributes, target_attr, 4, False, 0, False, True)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# discretizando los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data8 = copy.deepcopy(data)
data8 = utils.process_missing_values(data8,attributes, False)
data8 = utils.process_numeric_values_discretize(data8,attributes)
result8 = utils.cross_validation(data8, attributes, target_attr, 4, False, 0, False, True)


print("-------------------------------------------------------------------------------------")
print("Naive Bayes")
print()
# print("Tasa de aciertos sobre un conjunto de validación con", len(validation_set), "ejemplos:")
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
print("-------------------------------------------------------------------------------------")
