import glob

import pandas as pd
import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anomaliesDetection import AnomaliesDetection
from naive_bayes import NaiveBayes
import scipy.stats as st
import copy
import ejer2
import KNN
from sklearn.feature_selection import chi2

from scipy.stats import norm
import matplotlib.pyplot as plt

#####################################################################
#                          Constantes                               #
#####################################################################

all_files = ['Health-Tweets/bbchealth.txt',
             'Health-Tweets/cbchealth.txt',
             'Health-Tweets/cnnhealth.txt',
             'Health-Tweets/everydayhealth.txt',
             'Health-Tweets/foxnewshealth.txt',
             'Health-Tweets/gdnhealthcare.txt',
             'Health-Tweets/goodhealth.txt',
             'Health-Tweets/KaiserHealthNews.txt',
             'Health-Tweets/latimeshealth.txt',
             'Health-Tweets/msnhealthnews.txt',
             'Health-Tweets/NBChealth.txt',
             'Health-Tweets/nprhealth.txt',
             'Health-Tweets/nytimeshealth.txt',
             'Health-Tweets/reuters_health.txt',
             'Health-Tweets/usnewshealth.txt',
             'Health-Tweets/wsjhealth.txt']


#####################################################################
#         Importación y preprocesamiento de los tweets              #
#####################################################################
print('Leyendo conjunto de tweets...')
df = pd.concat(
    (pd.read_csv(
     f, sep='|', names=['Id', 'Date', 'Message'], encoding = "ISO-8859-1") for f in all_files))
print('Lectura finalizada')

# Nos quedamos solo con la columna de mensajes.
tweets = df['Message']

# Utilizamos CountVectorizer para realizar la tokenización y conteo.
# Adicionalmente se le pasa como parámetro una función preprocessor que
# preprocesará cada tweet antes de la tokenización.
# cv = CountVectorizer(stop_words='english',
#                      preprocessor=utils.preprocess_tweets)
# matrix = cv.fit_transform(tweets).toarray()


# print(cv.vocabulary_)
# print("-------")
# print(matrix)
# print(cv.get_feature_names())
# print("--------")
# print(len(matrix))

#####################################################################
######################### Ejercicio 2 ###############################
#####################################################################
print('-----------------------------------------------------------------------')
print()
print("Ejercicio 2")
print()

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

n = 5

categorical_atts_indexes = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]
non_categorical_atts_indexes = [10]

data = utils.process_missing_values(data_set, attributes, True)
data = ejer2.decode_data(data)

# Se divide el conjunto de datos
data_20, data_80 = utils.split_20_80(data)

# Se arman los training y validation set 
training_data, training_target_attributes = ejer2.extract_target_attributes(data_80)

numeric_training_set = ejer2.one_hot_encoding(training_data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)

validation_data, validation_target_attributes = ejer2.extract_target_attributes(data_20)
numeric_validation_set = ejer2.one_hot_encoding(validation_data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)

numeric_attributes = list(numeric_training_set[0].keys())
numeric_atts_len = len(numeric_attributes)
print('cantidad de atributos originales', len(categorical_atts))
print('cantidad de atributos luego de onehot encoding', numeric_atts_len)
print('tamaño del conjunto de validación', len(validation_data))



#######################################################################################
###########################             PCA            ################################
#######################################################################################
PC_training_data, PC_validation_data, PC_attributes = ejer2.PCA_validation_and_training_data(copy.deepcopy(numeric_training_set), copy.deepcopy(numeric_validation_set), 
                                                    copy.deepcopy(training_target_attributes), copy.deepcopy(validation_target_attributes), 
                                                    numeric_attributes, target_attr, n, 95)

                                                                            # (data, validation_set, target_attr, attributes, k, weight, normalize, use_standarization):
errors_KNN = KNN.holdout_validation(PC_training_data, PC_validation_data, target_attr, PC_attributes, 3, True, True, True)

print()
print('se utilizán los primeros', n, 'atributos de mayor importancia devueltos por CHI2')

print()
print('cantidad de errores KNN/PCA:',errors_KNN[1])

nb_classifier = NaiveBayes(PC_training_data, PC_attributes, target_attr)
errors_NB = nb_classifier.holdout_validation(PC_validation_data, target_attr)
print('cantidad de errores NB/PCA:',errors_NB[1])


#######################################################################################
###########################             CHI2           ################################
#######################################################################################

training_set_array = []
for x in copy.deepcopy(numeric_training_set):
    training_set_array.append(list(x.values()))

chi2_results = chi2(training_set_array, training_target_attributes)
chi2_result_list = chi2_results[0].tolist()

max_indexes = ejer2.get_n_max_indexes(chi2_result_list, n)
max_attributes = []
for j in max_indexes:
    max_attributes.append(numeric_attributes[j])

chi_training_set = ejer2.format_data_chi(max_attributes, copy.deepcopy(numeric_training_set))
chi_attributes = list(chi_training_set[0].keys())

ejer2.insert_target_attributes(chi_training_set, target_attr, training_target_attributes)

chi_validation_set = ejer2.format_data_chi(max_attributes, copy.deepcopy(numeric_validation_set))
ejer2.insert_target_attributes(chi_validation_set, target_attr, validation_target_attributes)

chi_errors_KNN = KNN.holdout_validation(chi_training_set, chi_validation_set, 
                                        target_attr, chi_attributes, 3, True, True, True)
print()
print('cantidad de errores KNN/CHI2:',chi_errors_KNN[1])

nb_classifier = NaiveBayes(chi_training_set, chi_attributes, target_attr)
chi_errors_NB = nb_classifier.holdout_validation(chi_validation_set, target_attr)
print('cantidad de errores NB/CHI2:',chi_errors_NB[1])
print()
print('-----------------------------------------------------------------------')



#####################################################################
######################### Ejercicio 3 ###############################
#####################################################################

#####################################################################
#################### DETECCION DE ANOMALIAS #########################
#####################################################################
print()
print("Ejercicio 3")
print()

file_names = ["bbchealth.txt", "cbchealth.txt", "cnnhealth.txt", "everydayhealth.txt", "gdnhealthcare.txt" , "goodhealth.txt",
			  "latimeshealth.txt", "nprhealth.txt", "nytimeshealth.txt", "reuters_health.txt", "usnewshealth.txt",
			  "foxnewshealth.txt", "KaiserHealthNews.txt", "msnhealthnews.txt", "NBChealth.txt", "wsjhealth.txt"]

tweetCollection = pd.concat((pd.read_csv("Health-Tweets/" + f, sep='|', encoding= "ISO-8859-1", names=['Id','Date','Message']) for f in file_names))

tweet = 'The boy love learn automatic learning in a hospital'

#tweet es la columna de mensajes
data = tweetCollection['Message']

#Se crea la instancia AnomaliesDetection donde se inicializa un countVectorizer
anomaliesDetector = AnomaliesDetection(data)

print("Se ejecutarán 3 mátodos para identificar un tweet anórmalos")
print("El tweet a analizar es el siguiente:")
print('        The boy love learn automatic learning in a hospital')
print()

print("Primer método retorna: ")
print(anomaliesDetector.firstMethod(data, tweet))
print()
print("Segundo método retorna:")
print(anomaliesDetector.secondMethod(data, tweet))
print()
print("Tercer método retorna:")
print(anomaliesDetector.thirdMethod(data, tweet))
