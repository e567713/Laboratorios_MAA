import utils
from naive_bayes import NaiveBayes


# Leemos data set del laboratorio
examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos


# Separamos el data set en dos subconjuntos
splitted_data = utils.split_20_80(data_set)

validation_set = splitted_data[0]
training_set = splitted_data[1]

utils.read_file

# Atributos a tener en cuenta
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

res = NaiveBayes(training_set, attributes, 'Class/ASD')
# res(instance) = clasificacion

# nv.clasify(insance)

# utils.v
values_frecuency = {}
print(values_frecuency)
print(res.values_frecuency)
for key, value in res.values_frecuency.items():
    print(key)
    print(value)
    for key2, value2 in value.items():
        print('   '+ key2)
print(res.target_values_frecuency[b'YES'])
# print(res.values_frecuency['jundice'])