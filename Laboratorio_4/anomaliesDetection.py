import math
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
        tweet = df['Message']
        # print(df)
        cv = CountVectorizer()
        x_traincvMatrix = cv.fit_transform(tweet)

        print(x_traincvMatrix.toarray())
        print(cv.get_feature_names())

        print()
        cv1 = CountVectorizer()
        x_traincv=cv1.fit_transform(tweet)
        a=x_traincv.toarray()
        # cv1.inverse_transform(a[0])
        # print(tweet.iloc[0])

        cv2 = TfidfVectorizer(min_df=1, stop_words='english')

        # tweet2= ['Hola','Chau','Hola Chau adios bebe','Hello']
        x_traincv2 = cv2.fit_transform(["Hi How are you How are you doing blue","Hi blue blue blue blue what's up","Wow that's awesome"])
        # x_traincv2 = cv2.fit_transform(tweet2)
        print(cv2.get_feature_names())
        print(x_traincv2.toarray())



        df['length'] = df['Message'].apply(len)
        print(df['length'].describe())
        print(df[df['length'] == 58]['Message'].iloc[0])

    def normal_probability(self, value, media , variance):
        return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

