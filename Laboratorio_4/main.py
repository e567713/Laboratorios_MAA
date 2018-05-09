import utils
from anomaliesDetection import AnomaliesDetection
import math
from naive_bayes import NaiveBayes


def normal_probability(self, value, media , variance):
    return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

name = 'bbchealth.txt'
# nb_classifier_theoric = AnomaliesDetection(name)
examples = utils.read_file('EjemploBayesNumerico.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos

attributes_theoric = ['Num1',
                      'Num2',
                      'Num3',
                      'Num4']

target_attr_theoric = 'Resultado'

nb_classifier_theoric = NaiveBayes(data_set, attributes_theoric, target_attr_theoric)
for instance in data_set:
    print(nb_classifier_theoric.classify(instance))

