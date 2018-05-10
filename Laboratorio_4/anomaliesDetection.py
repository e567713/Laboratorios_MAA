import math
import numpy as np
import scipy.stats as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB

class AnomaliesDetection:
    def __init__(self, name):

        # self.values_frecuency = {0:[], 1:[], 2:[]}
        # for line in open(name):
        #     fields = line.split('|')
        #     for i in range(3):
        #         self.values_frecuency[i].append(fields[i])

        df=pd.read_csv(name,sep='|',names=['Id','Date','Message'])

        #tweet es la columna de mensajes
        tweet = df['Message']

        #inicializo el CountVectorize quitando las stopword
        # cv = CountVectorizer(stop_words='english')

        #creo matriz de palabras y tweets
        # x_traincvMatrix = cv.fit_transform(tweet)

        # print(x_traincvMatrix.toarray()[0])
        # print(2)

        # cv.inverse_transform(x_traincvMatrix.toarray()[0])
        # print(3)
        #
        # print(tweet.iloc[0])
        # print(4)

        # print(x_traincvMatrix.toarray()[0])
        # cv = TfidfVectorizer(min_df=1, stop_words='english')


        #defino df['largo'] como el largo de los mensajes
        # df['largo'] = df['Message'].apply(len)

        #imprime caracteristicas
        # print(df['largo'].describe())

        #obtiene los de largo 58
        # print(df[df['largo'] == 58]['Message'].iloc[0])


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
        # mean = 0
        # std_dev = 1
        # dist = st.norm(loc=mean,scale=std_dev)
        # print(matrix[0])
        # dist.pdf(matrix[0])

def normal_probability(self, value, media , variance):
        return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

