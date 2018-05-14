import pandas as pd
import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anomaliesDetection import AnomaliesDetection
from naive_bayes import NaiveBayes


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
cv = CountVectorizer(stop_words='english',
                     preprocessor=utils.preprocess_tweets)
matrix = cv.fit_transform(tweets).toarray()


# print(cv.vocabulary_)
# print("-------")
# print(matrix)
# print(cv.get_feature_names())
# print("--------")
# print(len(matrix))


#####################################################################
######################### Ejercicio 3 ###############################
#####################################################################


# Calculamos media y varianza de cada feature
# means = []
# variances = []
# for column in range(matrix.shape[1]):
# 	means.append(np.mean(matrix[:,column]))
# 	variances.append(np.var(matrix[:,column]))

# instance = cv.transform(["Ernesto Fernandez Ferreyra"]).toarray()[0]

# print(instance)

# prob = 1
# for column in range(len(instance)):
# 	print(means[column])
# 	print(variances[column])
# 	prob *= utils.normal_probability(instance[column], means[column], variances[column])

# print(prob)
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
