import math
import numpy as np
import utils
import scipy.stats as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class AnomaliesDetection:
    def __init__(self, data):

        self.meansThird = []
        self.variancesThird = []
        self.countVectorizer = CountVectorizer(stop_words='english', min_df=300, preprocessor=utils.preprocess_tweets)

        self.matrix = self.countVectorizer.fit_transform(data).toarray()

        # Calculamos media y varianza de cada feature
        self.means = []
        self.variances = []

        for column in range(self.matrix.shape[1]):
            self.means.append(np.mean(self.matrix[:,column]))
            self.variances.append(np.var(self.matrix[:,column]))



    def firstMethod(self, data, tweet):

        instance = self.countVectorizer.transform([tweet]).toarray()

        prob= 1
        for column in range(self.matrix.shape[1]):
            dist = st.norm(loc=self.means[column],scale=math.sqrt(self.variances[column]))

            prob *= dist.pdf(      instance[0][column]    )

        return prob


    def secondMethod(self, data, tweet):

        instance = self.countVectorizer.transform([tweet]).toarray()

        prob= 1
        for column in range(self.matrix.shape[1]):
            dist = st.norm(loc=self.means[column],scale=math.sqrt(self.variances[column]))

            prob *= (dist.pdf( instance[0][column])  /  dist.pdf(self.means[column]))

        return prob

    def thirdMethod(self, data, tweet):


        sumWord = 0
        sumLetter = 0
        varWord = 0
        varLetter = 0

        #Se calcula la media
        for i in range(len(data)):
            sumWord+= len(utils.preprocess_tweets(data.iloc[i]).split())
            sumLetter+= len(utils.preprocess_tweets(data.iloc[i]))
        meanWord= sumWord /len(data)
        meanLetter= sumLetter /len(data)

        #Se calcula la varianza
        for o in  range(len(data)):
            varWord += (len(utils.preprocess_tweets(data.iloc[o]).split())-meanWord)**2
            varLetter += (len(utils.preprocess_tweets(data.iloc[o]))-meanLetter)**2
        varWord= (varWord/len(data))
        varLetter= (varLetter/len(data))

        probWord = utils.normal_probability2(len(tweet.split()), meanWord, math.sqrt(varWord))
        probLetter = utils.normal_probability2(len(tweet), meanLetter, math.sqrt(varLetter))

        return probWord *probLetter

