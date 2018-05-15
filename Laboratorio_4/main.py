import pandas as pd
import utils
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from anomaliesDetection import AnomaliesDetection
from naive_bayes import NaiveBayes
from K_Means import K_Means
from sklearn.cluster import KMeans


#####################################################################
#                          Constantes                               #
#####################################################################
# all_files = ['Health-Tweets/example.txt']

# all_files = ['Health-Tweets/example.txt',
# 			 'Health-Tweets/otherexample.txt']


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


# Constantes para CountVectorizer
min_df = 1000


# Constantes para correr K_Means

# Número de clusters.
k = 10 

# Máximo de iteraciones en cada corrida del algoritmo (si no se converge antes).
max_iter = 5 

# Número de inicializaciones distintas a realizar.
times = 1

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
cv = CountVectorizer(stop_words='english', min_df=min_df,
                     preprocessor=utils.preprocess_tweets)
matrix = cv.fit_transform(tweets).toarray()


# print(cv.vocabulary_)
# print("-------")
# print(list(matrix))
# print(cv.get_feature_names())
# print("--------")
# print(len(matrix))
# names = cv.get_feature_names()

data = list(matrix)
print(len(data)) 
print('Quitando tweets vacíos...')
cant = 0
for i,twe in enumerate(data):
	if all(v == 0 for v in twe):
		del data[i]
		cant += 1
print('Se termina de quitar tweets vacíos')
print(len(data)) 


#####################################################################
######################### Ejercicio 1 ###############################
#####################################################################

kmeans = K_Means(k, max_iter, times)
kmeans.train(data)
print(kmeans.optimal[0])
scikit_kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iter, n_init=times).fit(data)
print(scikit_kmeans.cluster_centers_)

for centroid in scikit_kmeans.cluster_centers_:
	for coord in centroid:
		print(round(coord, 3))
file = open('results.txt', 'a')
file.write('-----------------------------------------------------')
file.write('\n')
file.write('ALGORITMO SCIKIT K-Means ') 
file.write('\n')
file.write('\n')
file.write('Los centroides hallados scikit son: ')
file.write('\n')
file.write('\n')
for centroid in scikit_kmeans.cluster_centers_:
	for coord in centroid:
		file.write(str(round(coord, 3)))
		file.write(' , ')
	file.write('\n')
	file.write('\n')
	file.write('*******************************')
	file.write('\n')
	file.write('\n')
file.close()

diff = []
for index, centroid in enumerate(kmeans.optimal[0]):
	diff.append(kmeans.euclidean_distance(centroid, scikit_kmeans.cluster_centers_[index]))

print(diff)
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
