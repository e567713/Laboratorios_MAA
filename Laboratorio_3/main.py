# coding=utf-8
import utils
import id3
from naive_bayes import NaiveBayes
import KNN
import copy
import numpy as np


# Atributos del ejemplo teórico a tener en cuenta
target_attr_theoric = 'Juega'
attributes_theoric = ['Tiempo',
              'Temperatura',
              'Humedad',
              'Viento']

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
# Leemos data set del teórico
examples_theoric = utils.read_file('Theoric.arff')
data_set_theoric = copy.deepcopy(examples_theoric[0])  # Datos
metadata_theoric = examples_theoric[1]  # Metadatos


# Se procesan los valores faltantes
# utils.process_missing_values(data_set,attributes, True)


# Separamos el data set en dos subconjuntos
# splitted_data = utils.split_20_80(data_set)
#
# validation_set = splitted_data[0]
# training_set = splitted_data[1]
#


#######################################################################################
###########################           Parte 1          ################################
#######################################################################################

instance = {'Tiempo': 'Soleado', 'Temperatura': 'Frio', 'Humedad': 'Alta','Viento': 'Fuerte'}
instance2 = {'Tiempo': b'Soleado', 'Temperatura': b'Frio', 'Humedad': b'Alta','Viento': b'Fuerte'}
print ()
print ('Parte A)' )
print("Para el ejemplo del teórico se clasifica la instancia <Soleado, Frio, Alta, Fuerte>")
print()
print()

#Se clasifica la instancia con NaiveBayes

# Entrena 
decoded_data_set = utils.decode_set(data_set_theoric)
nb_classifier_theoric = NaiveBayes(data_set_theoric, attributes_theoric, target_attr_theoric)

# Clasifica
result_nb_theoric = nb_classifier_theoric.classify(instance2,False)

print ("Naive Bayes clasifica la instancia como: ")
print("\t",result_nb_theoric)
print ()

# Se clasifica la instancia con ID3

# Entrena
tree = id3.ID3_algorithm(data_set_theoric, attributes_theoric, 'Juega', True, False)

# Clasifica
result_id3_theoric = id3.validate_instance_extended(tree, instance2, b'Juega').decode()

print ("ID3 clasifica la instancia como: ")
print("\t",result_id3_theoric)
print ()

# Se clasifica la instancia con KNN

# Clasifica
result_knn_1_theoric = KNN.classify(instance, data_set_theoric, 1, 'Juega', True, attributes_theoric).decode()
result_knn_3_theoric = KNN.classify(instance, data_set_theoric, 3, 'Juega', True, attributes_theoric).decode()
result_knn_7_theoric = KNN.classify(instance, data_set_theoric, 7, 'Juega', True, attributes_theoric).decode()

print ("KNN clasifica la instancia como: ")
print('k = 1 ---> ',result_knn_1_theoric)
print('k = 3 ---> ',result_knn_3_theoric)
print('k = 7 ---> ',result_knn_7_theoric)
print ()

print("-------------------------------------------------------------------------------------")
print()
print('Parte C) 1.')
print('Hay 8 variantes del algoritmo Naive Bayes')
print('Se ejecutará 10-fold cross-validation para cada una')


# Se copia el conjunto de datos original para no alterarlo.
data = copy.deepcopy(data_set)
# Se ordenan aleatoriamente los ejemplos del conjunto para simular la
# elección al azar de elementos para formar los subconjuntos.
np.random.shuffle(data)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# sin discretizar los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data1 = copy.deepcopy(data)
data1 = utils.process_missing_values(data1,attributes, True)
result1 = utils.cross_validation(data1, attributes, target_attr, 10, False, None, None, False, None)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# sin discretizar los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data2 = copy.deepcopy(data)
data2 = utils.process_missing_values(data2,attributes, False)
result2 = utils.cross_validation(data2, attributes, target_attr, 10, False, None, None, False, None)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# discretizando los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data3 = copy.deepcopy(data)
data3 = utils.process_missing_values(data3,attributes, True)
data3 = utils.process_numeric_values_discretize(data3,attributes)
result3 = utils.cross_validation(data3, attributes, target_attr, 10, False, None, None, False, None)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# discretizando los valores numéricos y sin calcular la probabilidad condicionada de valores numéricos como normal.
data4 = copy.deepcopy(data)
data4 = utils.process_missing_values(data4,attributes, False)
data4 = utils.process_numeric_values_discretize(data4,attributes)
result4 = utils.cross_validation(data4, attributes, target_attr, 10, False, None, None, False, None)



# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# sin discretizar los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data5 = copy.deepcopy(data)
data5 = utils.process_missing_values(data5,attributes, True)
result5 = utils.cross_validation(data5, attributes, target_attr, 10, False, None, None, True, None)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# sin discretizar los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data6 = copy.deepcopy(data)
data6 = utils.process_missing_values(data6,attributes, False)
result6 = utils.cross_validation(data6, attributes, target_attr, 10, False, None, None, True, None)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo tomando los valores faltantes como más comunes,
# discretizando los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data7 = copy.deepcopy(data)
data7 = utils.process_missing_values(data7,attributes, True)
data7 = utils.process_numeric_values_discretize(data7,attributes)
result7 = utils.cross_validation(data7, attributes, target_attr, 10, False, None, None, True, None)

# Se realiza una cross_validation con el algoritmo bayesiano sencillo eliminando la instancia con valores faltantes,
# discretizando los valores numéricos y calculando la probabilidad condicionada de valores numéricos como normal.
data8 = copy.deepcopy(data)
data8 = utils.process_missing_values(data8,attributes, False)
data8 = utils.process_numeric_values_discretize(data8,attributes)
result8 = utils.cross_validation(data8, attributes, target_attr, 10, False, None, None, True, None)


print("-------------------------------------------------------------------------------------")
print("Naive Bayes")
print()
# print("Tasa de aciertos sobre un conjunto de validación con", len(validation_set), "ejemplos:")
print("Tasa de aciertos sobre un conjunto de validación:")
print ("result 1:", 1-result1)
print ("result 2:", 1-result2)
print ("result 3:", 1-result3)
print ("result 4:", 1-result4)
print ("result 5:", 1-result5)
print ("result 6:", 1-result6)
print ("result 7:", 1-result7)
print ("result 8:", 1-result8)
print("-------------------------------------------------------------------------------------")



#######################################################################################
###########################             KNN            ################################
#######################################################################################

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
print('Se ejecutará 10-fold cross-validation para cada una')
print('Cada ejecución lleva su tiempo...')
print()


#######################################################################################
###########################            K = 1           ################################
#######################################################################################


print('1) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data1 = copy.deepcopy(splitted_data[1])
data1 = utils.process_missing_values(data1, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result1 = utils.cross_validation(data1, attributes, 'Class/ASD', 10, True, 1, True, True, True)



print('2) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data2 = copy.deepcopy(splitted_data[1])
data2 = utils.process_missing_values(data2, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result2 = utils.cross_validation(data2, attributes, 'Class/ASD', 10, True, 1, True, True, False)



print('3) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data3 = copy.deepcopy(splitted_data[1])
data3 = utils.process_missing_values(data3, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result3 = utils.cross_validation(data3, attributes, 'Class/ASD', 10, True, 1, False, True, True)



print('4) Cross-Validation KNN con k = 1, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data4 = copy.deepcopy(splitted_data[1])
data4 = utils.process_missing_values(data4, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result4 = utils.cross_validation(data4, attributes, 'Class/ASD', 10, True, 1, False, True, False)



print('5) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data5 = copy.deepcopy(splitted_data[1])
data5 = utils.process_missing_values(data5, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result5 = utils.cross_validation(data5, attributes, 'Class/ASD', 10, True, 1, False, True, True)



print('6) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data6 = copy.deepcopy(splitted_data[1])
data6 = utils.process_missing_values(data6, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result6 = utils.cross_validation(data6, attributes, 'Class/ASD', 10, True, 1, True, True, False)



print('7) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data7 = copy.deepcopy(splitted_data[1])
data7 = utils.process_missing_values(data7, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result7 = utils.cross_validation(data7, attributes, 'Class/ASD', 10, True, 1, False, True, True)



print('8) Cross-Validation KNN con k = 1, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data8 = copy.deepcopy(splitted_data[1])
data8 = utils.process_missing_values(data8, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result8 = utils.cross_validation(data8, attributes, 'Class/ASD', 10, True, 1, False, True, False)


#######################################################################################
###########################            K = 3           ################################
#######################################################################################


print('9) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data9 = copy.deepcopy(splitted_data[1])
data9 = utils.process_missing_values(data9, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result9 = utils.cross_validation(data9, attributes, 'Class/ASD', 10, True, 3, True, True, True)



print('10) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data10 = copy.deepcopy(splitted_data[1])
data10 = utils.process_missing_values(data10, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result10 = utils.cross_validation(data10, attributes, 'Class/ASD', 10, True, 3, True, True, False)



print('11) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data11 = copy.deepcopy(splitted_data[1])
data11 = utils.process_missing_values(data11, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result11 = utils.cross_validation(data11, attributes, 'Class/ASD', 10, True, 3, False, True, True)



print('12) Cross-Validation KNN con k = 3, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data12 = copy.deepcopy(splitted_data[1])
data12 = utils.process_missing_values(data12, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result12 = utils.cross_validation(data12, attributes, 'Class/ASD', 10, True, 3, False, True, False)



print('13) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data13 = copy.deepcopy(splitted_data[1])
data13 = utils.process_missing_values(data13, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result13 = utils.cross_validation(data13, attributes, 'Class/ASD', 10, True, 3, False, True, True)



print('14) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data14 = copy.deepcopy(splitted_data[1])
data14 = utils.process_missing_values(data14, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result14 = utils.cross_validation(data14, attributes, 'Class/ASD', 10, True, 3, True, True, False)



print('15) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data15 = copy.deepcopy(splitted_data[1])
data15 = utils.process_missing_values(data15, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result15 = utils.cross_validation(data15, attributes, 'Class/ASD', 10, True, 3, False, True, True)



print('16) Cross-Validation KNN con k = 3, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data16 = copy.deepcopy(splitted_data[1])
data16 = utils.process_missing_values(data16, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result16 = utils.cross_validation(data16, attributes, 'Class/ASD', 10, True, 3, False, True, False)


#######################################################################################
###########################            K = 7           ################################
#######################################################################################


print('17) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data17 = copy.deepcopy(splitted_data[1])
data17 = utils.process_missing_values(data17, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result17 = utils.cross_validation(data17, attributes, 'Class/ASD', 10, True, 7, True, True, True)



print('18) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data18 = copy.deepcopy(splitted_data[1])
data18 = utils.process_missing_values(data18, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result18 = utils.cross_validation(data18, attributes, 'Class/ASD', 10, True, 7, True, True, False)



print('19) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data19 = copy.deepcopy(splitted_data[1])
data19 = utils.process_missing_values(data19, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result19 = utils.cross_validation(data19, attributes, 'Class/ASD', 10, True, 7, False, True, True)



print('20) Cross-Validation KNN con k = 7, usa most-common para atributos faltantes, NO usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data20 = copy.deepcopy(splitted_data[1])
data20 = utils.process_missing_values(data20, attributes, True)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result20 = utils.cross_validation(data20, attributes, 'Class/ASD', 10, True, 7, False, True, False)



print('21) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data21 = copy.deepcopy(splitted_data[1])
data21 = utils.process_missing_values(data21, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result21 = utils.cross_validation(data21, attributes, 'Class/ASD', 10, True, 7, False, True, True)



print('22) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data22 = copy.deepcopy(splitted_data[1])
data22 = utils.process_missing_values(data22, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result22 = utils.cross_validation(data22, attributes, 'Class/ASD', 10, True, 7, True, True, False)



print('23) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando standarization')

# Me quedo con el 80% del conjunto
data23 = copy.deepcopy(splitted_data[1])
data23 = utils.process_missing_values(data23, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result23 = utils.cross_validation(data23, attributes, 'Class/ASD', 10, True, 7, False, True, True)



print('24) Cross-Validation KNN con k = 7, descarta ejemplos con atributos faltantes, NO usa pesos y normaliza utilizando min-max')

# Me quedo con el 80% del conjunto
data24 = copy.deepcopy(splitted_data[1])
data24 = utils.process_missing_values(data24, attributes, False)
            # cross_validation(data, attributes, target_attr, k_fold, applicate_KNN, k, weight, normalize, use_standarization)
result24 = utils.cross_validation(data24, attributes, 'Class/ASD', 10, True, 7, False, True, False)


print("-------------------------------------------------------------------------------------")
print("KNN")
print()
print("Tasa de aciertos sobre un conjunto de validación:")
print ("result 1:", 1-result1)
print ("result 2:", 1-result2)
print ("result 3:", 1-result3)
print ("result 4:", 1-result4)
print ("result 5:", 1-result5)
print ("result 6:", 1-result6)
print ("result 7:", 1-result7)
print ("result 8:", 1-result8)
print ("result 9:", 1-result9)
print ("result 10:", 1-result10)
print ("result 11:", 1-result11)
print ("result 12:", 1-result12)
print ("result 13:", 1-result13)
print ("result 14:", 1-result14)
print ("result 15:", 1-result15)
print ("result 16:", 1-result16)
print ("result 17:", 1-result17)
print ("result 18:", 1-result18)
print ("result 19:", 1-result19)
print ("result 20:", 1-result20)
print ("result 21:", 1-result21)
print ("result 22:", 1-result22)
print ("result 23:", 1-result23)
print ("result 24:", 1-result24)
print("-------------------------------------------------------------------------------------")


results = [result1, result2, result3, result4, result5, result6, result7, result8, result9, 
            result10, result11, result12, result13, result14, result15, result16, result17, 
            result18, result19, result20, result21, result22, result23, result24]

parameters = [[True, 1, True, True], [True, 1, True, False], [True, 1, False, True], [True, 1, False, False], [False, 1, True, True], [False, 1, True, False], [False, 1, False, True], [False, 1, False, False],
              [True, 3, True, True], [True, 3, True, False], [True, 3, False, True], [True, 3, False, False], [False, 3, True, True], [False, 3, True, False], [False, 3, False, True], [False, 3, False, False],
              [True, 7, True, True], [True, 7, True, False], [True, 7, False, True], [True, 7, False, False], [False, 7, True, True], [False, 7, True, False], [False, 7, False, True], [False, 7, False, False]]

print("-------------------------------------------------------------------------------------")
print()
print('Parte C) 2.')