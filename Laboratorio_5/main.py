import sys

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


minCost = float('inf')
for w in range(4):
    for a in range(4):
        weight= []
        # Weights para LR
        for i in range(len(LR_numeric_attributes)):
            weight += [posiblesWeight[w]]

        # Constante alpha de LR
        alpha = posiblesAlpha[a]

        for i in range(10):
            # Se ajusta Weight con decenso por gradiente 10 veces
            weight = utils.descentByGradient(weight, LR_training_set_scaled, alpha, LR_numeric_attributes, target_attr)

        cost = utils.costFunction(weight, LR_training_set_scaled, LR_numeric_attributes, target_attr)

        if (minCost > cost):
            minCost = cost
            bestChoose = [posiblesWeight[w], posiblesAlpha[a], cost]

print("El menor costo conseguido fue: ", bestChoose[2], " con Weight = ", bestChoose[0], " y alpha = ", bestChoose[1])


#######################################################################################
################       Se calculan los errores para los            ####################
################                 algoritmos                        ####################
#######################################################################################

errors_LR_total = 0
errors_KNN_total = 0
errors_NB_total = 0
errors_ID3_total = 0

for iter in range(10):
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
        weight += [bestChose[0]]

    # Constante alpha de LR
    alpha = bestChose[1]

    # Costo anterior
    cost = sys.maxsize

    for i in range(100):
        newCost = utils.costFunction(weight, LR_training_set_scaled, LR_numeric_attributes, target_attr)
        dif =abs(cost - newCost)
        if (abs(cost - newCost) < 0.0001):
            break
        print("Costo", i + 1)
        print(newCost)
        print()
        cost = newCost
        weight = utils.descentByGradient(weight, LR_training_set_scaled, alpha, LR_numeric_attributes, target_attr)


    # LR holdout validation
    errors_LR = utils.LR_holdout_validation(LR_validation_set_scaled, target_attr, weight, LR_numeric_attributes)
    # print("Cantidad de errores registrados LR:", errors_LR)
    errors_LR_total += errors_LR


    # KNN holdout validation con k = 3 y usando pesos
    errors_KNN = utils.KNN_holdout_validation(copy.deepcopy(training_set_scaled), copy.deepcopy(validation_set_scaled), target_attr, numeric_attributes, 3, True)
    # print('cantidad de errores KNN:',errors_KNN[1])
    errors_KNN_total += errors_KNN[1]

    # Para naive bayes hay que ver si pasarle el scaled o no,
    # se rompe con el scaled, habría que usar la normalizacion de naive bayes?

    nb_classifier = NaiveBayes(copy.deepcopy(training_set_scaled), numeric_attributes, target_attr)
    # print(nb_classifier.attributes_values)
    errors_NB = nb_classifier.holdout_validation(copy.deepcopy(validation_set_scaled), target_attr)
    # print('cantidad de errores NB:',errors_NB[1])
    errors_NB_total += errors_NB[1]

    tree = id3.ID3_algorithm(training_set_scaled, numeric_attributes, target_attr, True, False)
    errors_ID3 = id3.validation(tree, validation_set_scaled, target_attr)
    # print("cantidad de errores ID3:",errors_ID3)
    errors_ID3_total += errors_ID3

print("Promedio de errores:")
print("LR: ", errors_LR_total/10)
print("KNN: ", errors_KNN_total/10)
print("NB: ", errors_NB_total/10)
print("ID3: ", errors_ID3_total/10)