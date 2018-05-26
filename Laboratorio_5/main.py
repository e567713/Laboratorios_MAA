import utils

examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos

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

# weight = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
weight = []

categorical_atts = ['A1_Score',
                    'A2_Score',
                    'A3_Score',
                    'A4_Score',
                    'A5_Score',
                    'A6_Score',
                    'A7_Score',
                    'A8_Score',
                    'A9_Score',
                    'A10_Score',
                    'gender',
                    'ethnicity',
                    'jundice',
                    'austim',
                    'contry_of_res',
                    'used_app_before',
                    'age_desc',
                    'relation']

non_categorical_atts = ['age']

n = 5

categorical_atts_indexes = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]
non_categorical_atts_indexes = [10]

data = utils.process_missing_values(data_set, attributes, True)
data = utils.decode_data(data)

# Se divide el conjunto de datos
data_20, data_80 = utils.split_20_80(data)

# Se arman los training y validation set
training_data, training_target_attributes = utils.extract_target_attributes(data_80)



numeric_training_set = utils.one_hot_encoding(training_data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)
validation_data, validation_target_attributes = utils.extract_target_attributes(data_20)
# numeric_validation_set = utils.one_hot_encoding(validation_data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)

numeric_attributes = ['sesgo']
numeric_attributes += list(numeric_training_set[0].keys())

utils.insert_sesgo_one(numeric_training_set)
utils.insert_target_attributes(numeric_training_set, target_attr, training_target_attributes)

for i in range(len(numeric_attributes)):
    weight += [0.1]

print("Peso 1")
print(weight)
print("Costo 1")
print(utils.costFunction(weight, numeric_training_set, numeric_attributes, target_attr))
print()

weight2 = utils.descentByGradient(weight, numeric_training_set, 1, numeric_attributes, target_attr)
print("Peso 2")
print(weight2)
print("Costo 2")
# print(utils.costFunction(weight2, numeric_training_set, numeric_attributes, target_attr))
print()

weight3 = utils.descentByGradient(weight2, numeric_training_set, 1, numeric_attributes, target_attr)
print("Peso 3")
print(weight3)
print("Costo 3")
# print(utils.costFunction(weight3, numeric_training_set, numeric_attributes, target_attr))

# print()
#
# print(utils.costFunction(weight, numeric_training_set, numeric_attributes, target_attr))
# weight = utils.descentByGradient(weight, numeric_training_set, 1, numeric_attributes, target_attr)
# print(weight)
print()




# utils.insert_target_attributes(numeric_validation_set, target_attr, validation_target_attributes)
#
# utils.insert_sesgo_one(numeric_validation_set, target_attr, validation_target_attributes)
#
# # numeric_attributes = list(numeric_training_set[0].keys())
# numeric_attributes = list(numeric_validation_set[0].keys())
# numeric_atts_len = len(numeric_attributes)
#
#
#
#
#





