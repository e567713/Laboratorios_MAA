import utils
from anomaliesDetection import AnomaliesDetection
import math


def normal_probability(self, value, media , variance):
    return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

name = 'bbchealth.txt'
nb_classifier_theoric = AnomaliesDetection(name)

