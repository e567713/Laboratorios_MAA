import glob

import pandas as pd
import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anomaliesDetection import AnomaliesDetection
from naive_bayes import NaiveBayes
import scipy.stats as st

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
