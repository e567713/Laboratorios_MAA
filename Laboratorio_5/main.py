import sys
from tabulate import tabulate
import utils
import copy
from naive_bayes import NaiveBayes
import id3


examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos

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

weight = []

categorical_atts = ['A1_Score',
                    'A2_Score',
                    'A3_Score',
                    'A4_Score',
                    'A5_Score',
                    'A6_Score',
                    'A7_Score',
                    'A8_Score',
                    'A9_Score',
                    'A10_Score',
                    'gender',
                    'ethnicity',
                    'jundice',
                    'austim',
                    'contry_of_res',
                    'used_app_before',
                    'age_desc',
                    'relation']

non_categorical_atts = ['age']


categorical_atts_indexes = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]
non_categorical_atts_indexes = [10]

# Se usa most-common para manejar los missing values
data = utils.process_missing_values(data_set, attributes, True)
# Decode bytes
data = utils.decode_data(data)

# Sacamos el target attribute
data_ext, data_target_attributes = utils.extract_target_attributes(data)

# one hot encoding
numeric_data = utils.one_hot_encoding(data_ext, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)
numeric_attributes = list(numeric_data[0].keys())

# insertamos target attribute
utils.insert_target_attributes(numeric_data, target_attr, data_target_attributes)

# 'numeric_data' es el 100% de los datos desp de realizar one hot encoding

#######################################################################################
################       Se buscan el mejor alpha y los mejores      ####################
################                 pesos iniciales                   ####################
#######################################################################################

#  Se divide el conjunto de datos
numeric_validation_set, numeric_training_set = utils.split_20_80(numeric_data)

bestChoose = []
posiblesWeight = [10, 1, 0.1, 0]
posiblesAlpha = [50, 10, 1, 0.5]
use_standarization = False


# Normalizamos el training set utilizando la tecnica min-max
training_set_scaled, scalation_parameters = utils.scale(copy.deepcopy(numeric_training_set), numeric_attributes,use_standarization)

# Normalizamos el validation set utilizando la tecnica min-max y
# los valores que usamos para normalizar el training
validation_set_scaled = []
for instance in numeric_validation_set:
    scaled_instance = utils.scale_instance(copy.deepcopy(instance), scalation_parameters, use_standarization)
    validation_set_scaled.append(scaled_instance)

# Sesgo para LR
LR_numeric_attributes = copy.deepcopy(numeric_attributes)
LR_numeric_attributes.insert(0,'sesgo')
LR_training_set_scaled = copy.deepcopy(training_set_scaled)
utils.insert_sesgo_one(LR_training_set_scaled)

print("Se realizan pruebas para obtiener los parametros con mejores resultados:")
tablaParametros = []
tablaParametros.append(["Weight", "Alpha", "Costo obtenido"])
minCost = float('inf')
for w in range(4):
    for a in range(4):
        weight= []
        # Weights para LR
        for i in range(len(LR_numeric_attributes)):
            weight += [posiblesWeight[w]]

        # Constante alpha de LR
        alpha = posiblesAlpha[a]

        for i in range(4):
            # Se ajusta Weight con decenso por gradiente 10 veces
            weight = utils.descentByGradient(weight, LR_training_set_scaled, alpha, LR_numeric_attributes, target_attr)

        cost = utils.costFunction(weight, LR_training_set_scaled, LR_numeric_attributes, target_attr)
        tablaParametros.append([posiblesWeight[w], alpha, cost])

        if (minCost > cost):
            minCost = cost
            bestChoose = [posiblesWeight[w], posiblesAlpha[a], cost]


print(tabulate(tablaParametros, headers='firstrow', tablefmt='fancy_grid', stralign='right', showindex=True))
print("El menor costo conseguido fue: ", bestChoose[2], " con Weight = ", bestChoose[0], " y alpha = ", bestChoose[1])
print("A continuación se utilizarán los parametros obtenidos.")
print()

#######################################################################################
################       Se calculan los errores para los            ####################
################                 algoritmos                        ####################
#######################################################################################

errors_LR_total = 0
errors_KNN_total = 0
errors_NB_total = 0
errors_ID3_total = 0
tablaIteraciones = []
tablaErroresIteraciones = []

tablaIteraciones.append(['','',"Iter 1","Iter 2","Iter 3","Iter 4","Iter 5","Iter 6","Iter 7","Iter 8"])
tablaErroresIteraciones.append(['','',"Iter 1","Iter 2","Iter 3","Iter 4","Iter 5","Iter 6","Iter 7","Iter 8","Promedio"])

for iter in range(8):
    print('Realizando iteración ',iter +1, '...')
    #######################################################################################
    #######################       Regresión Logica           ##############################
    #######################################################################################
    # Se divide el conjunto de datos
    numeric_validation_set, numeric_training_set = utils.split_20_80(numeric_data)


    use_standarization = False
    # Normalizamos el training set utilizando la tecnica min-max
    training_set_scaled, scalation_parameters = utils.scale(copy.deepcopy(numeric_training_set), numeric_attributes,use_standarization)


    # Normalizamos el validation set utilizando la tecnica min-max y
    # los valores que usamos para normalizar el training
    validation_set_scaled = []
    for instance in numeric_validation_set:
        scaled_instance = utils.scale_instance(copy.deepcopy(instance), scalation_parameters, use_standarization)
        validation_set_scaled.append(scaled_instance)


    # Sesgo para LR
    LR_numeric_attributes = copy.deepcopy(numeric_attributes)
    LR_numeric_attributes.insert(0,'sesgo')
    LR_training_set_scaled = copy.deepcopy(training_set_scaled)
    LR_validation_set_scaled = copy.deepcopy(validation_set_scaled)
    utils.insert_sesgo_one(LR_training_set_scaled)
    utils.insert_sesgo_one(LR_validation_set_scaled)

    # Weights para LR
    weight = []
    for i in range(len(LR_numeric_attributes)):
        weight += [bestChoose[0]]

    # Constante alpha de LR
    alpha = bestChoose[1]

    # Costo anterior
    cost = float('inf')

    # La condición de parado son 15 iteraciones o una
    # diferencia de costos menor a 0.0001
    for i in range(15):
        newCost = utils.costFunction(weight, LR_training_set_scaled, LR_numeric_attributes, target_attr)
        dif =abs(cost - newCost)

        if (iter == 0):
            row = 'Ajuste ' + repr(i)
            tablaIteraciones.append([row])

        tablaIteraciones[i+1].append(newCost)

        if (abs(cost - newCost) < 0.0001):
            break
        cost = newCost
        weight = utils.descentByGradient(weight, LR_training_set_scaled, alpha, LR_numeric_attributes, target_attr)


    # LR holdout validation
    errors_LR = utils.LR_holdout_validation(LR_validation_set_scaled, target_attr, weight, LR_numeric_attributes)
    errors_LR_total += errors_LR


    #######################################################################################
    #######################               KNN                ##############################
    #######################################################################################

    # KNN holdout validation con k = 3 y usando pesos
    errors_KNN = utils.KNN_holdout_validation(copy.deepcopy(training_set_scaled), copy.deepcopy(validation_set_scaled), target_attr, numeric_attributes, 3, True)
    errors_KNN_total += errors_KNN[1]


    #######################################################################################
    #######################         Naive Bayes              ##############################
    #######################################################################################

    nb_classifier = NaiveBayes(copy.deepcopy(training_set_scaled), numeric_attributes, target_attr)
    errors_NB = nb_classifier.holdout_validation(copy.deepcopy(validation_set_scaled), target_attr)
    errors_NB_total += errors_NB[1]


    #######################################################################################
    #######################            ID3                   ##############################
    #######################################################################################

    tree = id3.ID3_algorithm(training_set_scaled, numeric_attributes, target_attr, True, False)
    errors_ID3 = id3.validation(tree, validation_set_scaled, target_attr)
    errors_ID3_total += errors_ID3


    if (iter == 0):
        tablaErroresIteraciones.append(['Regresión lógica'])
        tablaErroresIteraciones.append(['KNN'])
        tablaErroresIteraciones.append(['Naive Bayes'])
        tablaErroresIteraciones.append(['ID3'])

    tablaErroresIteraciones[1].append(errors_LR)
    tablaErroresIteraciones[2].append(errors_KNN[1])
    tablaErroresIteraciones[3].append(errors_NB[1])
    tablaErroresIteraciones[4].append(errors_ID3)

tablaErroresIteraciones[1].append(errors_LR_total/8)
tablaErroresIteraciones[2].append(errors_KNN_total/8)
tablaErroresIteraciones[3].append(errors_NB_total/8)
tablaErroresIteraciones[4].append(errors_ID3_total/8)



print(tabulate(tablaIteraciones, headers='firstrow', tablefmt='fancy_grid', stralign='right', showindex=True))
print()
print(tabulate(tablaErroresIteraciones, headers='firstrow', tablefmt='fancy_grid', stralign='right', showindex=True))
