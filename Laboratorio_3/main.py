import utils
from naive_bayes import NaiveBayes

# Atributos a tener en cuenta
target_attr = 'Class/ASD'
attributes = ['A1_Score',
              'A2_Score',
              'A3_Score',
              'A4_Score',
              'A5_Score',
              'A6_Score',
              'A7_Score',
              'A8_Score',
              'A9_Score',
              'A10_Score',
              'age',
              'gender',
              'ethnicity',
              'jundice',
              'austim',
              'contry_of_res',
              'used_app_before',
              'age_desc',
              'relation']


#######################################################################################
################           Carga y procesamiento de datos          ####################
#######################################################################################
 
# Leemos data set del laboratorio
examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos

# Se procesan los valores faltantes
utils.process_missing_values(data_set,attributes)

# Se procesan los valores numéricos
# TODO
utils.process_numeric_values(data_set,attributes)

# Separamos el data set en dos subconjuntos
splitted_data = utils.split_20_80(data_set)

validation_set = splitted_data[0]
training_set = splitted_data[1]



#######################################################################################
###########################           Parte 1          ################################
#######################################################################################

# Se genera un clasificador bayesiano sencillo entrenándolo con el set de entrenamiento.
nb_classifier = NaiveBayes(training_set[:5], attributes, target_attr)

# Se valida el clasificador con el set de validación.
result = utils.validate(validation_set[:1] , nb_classifier, target_attr)

print(result)