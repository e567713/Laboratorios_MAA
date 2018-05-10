from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import utils
from anomaliesDetection import AnomaliesDetection
import math
from naive_bayes import NaiveBayes


def normal_probability(self, value, media , variance):
    return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

name = 'bbchealth.txt'
anomalyDetector = AnomaliesDetection(name)
# print(anomalyDetector)



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

df=pd.read_csv(name,sep='|',names=['Id','Date','Message'])

#tweet es la columna de mensajes
tweet = df['Message']


def removeLinks(s):
    return s.upper()

cv = CountVectorizer(stop_words='english', preprocessor=removeLinks())
matrix = cv.fit_transform(tweet)
print(cv.vocabulary_)
print("-------")
print(matrix.toarray())
print(cv.get_feature_names())
print("--------")

print(cv.transform(['Something papel new.']).toarray())
print(cv.vocabulary_.get('papel'))

transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(matrix.toarray())
print( tfidf.toarray() )
