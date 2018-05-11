import pandas as pd
import utils
import math
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anomaliesDetection import AnomaliesDetection
from naive_bayes import NaiveBayes


##################################################################### 
######################### Constantes ################################
##################################################################### 

file_names = 'bbchealth.txt' 


##################################################################### 
################# Importamos el conjunto de tweets ##################
##################################################################### 

df=pd.read_csv(file_names,sep='|',names=['Id','Date','Message'])

#tweet es la columna de mensajes
tweet = df['Message']

cv = CountVectorizer(stop_words='english', preprocessor=utils.preprocess_tweets)
matrix = cv.fit_transform(tweet).toarray()
print(cv.vocabulary_)
print("-------")
print(matrix)
print(cv.get_feature_names())
print("--------")



##################################################################### 
######################### Ejercicio 3 ###############################
##################################################################### 



# Calculamos media y varianza de cada feature
means = []
variances = []
for column in range(matrix.shape[1]):
	means.append(np.mean(matrix[:,column]))
	variances.append(np.var(matrix[:,column]))

instance = cv.transform(["Ernesto Fernandez Ferreyra"]).toarray()[0]

print(instance)

prob = 1
for column in range(len(instance)):
	print(means[column])
	print(variances[column])
	prob *= utils.normal_probability(instance[column], means[column], variances[column])

print(prob)
# anomalyDetector = AnomaliesDetection(file_names)



# examples = utils.read_file('EjemploBayesNumerico.arff')
# data_set = examples[0]  # Datos
# metadata = examples[1]  # Metadatos
#
# attributes_theoric = ['Num1',
#                       'Num2',
#                       'Num3',
#                       'Num4']
#
# target_attr_theoric = 'Resultado'
#
# nb_classifier_theoric = NaiveBayes(data_set, attributes_theoric, target_attr_theoric)
# for instance in data_set:
#     print(nb_classifier_theoric.classify(instance))



# cv = CountVectorizer(stop_words='english', preprocessor=removeLinks())
# matrix = cv.fit_transform(tweet)
# print(cv.vocabulary_)
# print("-------")
# print(matrix.toarray())
# print(cv.get_feature_names())
# print("--------")

# print(cv.transform(['Something papel new.']).toarray())
# print(cv.vocabulary_.get('papel'))

# transformer = TfidfTransformer(smooth_idf=False)
# tfidf = transformer.fit_transform(matrix.toarray())
# print( tfidf.toarray() )
