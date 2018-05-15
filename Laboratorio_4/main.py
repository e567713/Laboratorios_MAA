import glob

import pandas as pd
import utils
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anomaliesDetection import AnomaliesDetection
from naive_bayes import NaiveBayes
import scipy.stats as st

from scipy.stats import norm
import matplotlib.pyplot as plt

##################################################################### 
######################### Constantes ################################
##################################################################### 



##################################################################### 
################# Importamos el conjunto de tweets ##################
##################################################################### 


# vec = CountVectorizer(min_df=M).fit(tweet)
# bag_of_words = vec.transform(tweet)
# sum_words = bag_of_words.sum(axis=0)
# words_freq = [(word, sum_words[0, idx]) for word, idx in     vec.vocabulary_.items()]
# words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
# print(words_freq[:N])



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

tweet = 'Risk RT risk level bbc life VIDEO'

#tweet es la columna de mensajes
data = tweetCollection['Message']

#Se crea la instancia AnomaliesDetection donde se inicializa un countVectorizer
anomaliesDetector = AnomaliesDetection(data)

print("Se ejecutarán 3 mátodos para identificar tweet anórmalos")

print("Primer método retorna: ")
print(anomaliesDetector.firstMethod(data, tweet))
print()
print("Segundo método retorna:")
print(anomaliesDetector.secondMethod(data, tweet))
print()
print("Tercer método retorna:")
print(anomaliesDetector.thirdMethod(data, tweet))
