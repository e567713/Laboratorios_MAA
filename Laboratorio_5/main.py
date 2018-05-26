import utils
import copy

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

data_ext, data_target_attributes = utils.extract_target_attributes(data)

numeric_data = utils.one_hot_encoding(data_ext, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)
numeric_attributes = list(numeric_data[0].keys())

utils.insert_target_attributes(numeric_data, target_attr, data_target_attributes)

# Se divide el conjunto de datos
numeric_validation_set, numeric_training_set = utils.split_20_80(numeric_data)

training_set_scaled, scalation_parameters = utils.scale(copy.deepcopy(numeric_training_set), numeric_attributes,False)
validation_set_scaled, scalation_parameters2 = utils.scale(copy.deepcopy(numeric_validation_set), numeric_attributes,False)

# print()
# for x in training_data_scale[:3]: print(x)

numeric_attributes.insert(0,'sesgo')

utils.insert_sesgo_one(training_set_scaled)
utils.insert_sesgo_one(validation_set_scaled)


for i in range(len(numeric_attributes)):
    weight += [0.1]




alpha = 5


for i in range(25):
    # print("Peso", i + 1)
    # print(weight)
    print("Costo", i + 1)
    print(utils.costFunction(weight, training_set_scaled, numeric_attributes, target_attr))
    print()
    weight = utils.descentByGradient(weight, training_set_scaled, alpha, numeric_attributes, target_attr)


errores = 0
for instance in validation_set_scaled:
    result = utils.clssify_LR_instance(instance, weight, numeric_attributes)
    if (instance[target_attr]!=result):
        errores += 1
print("Cantidad de errores registrados:")
print(errores)


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





