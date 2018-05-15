import numpy as np
import scipy
import copy
import utils
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2
import KNN
from naive_bayes import NaiveBayes

def PCA(data, attributes, numeric_atts, cant_vectors, percentage):
  # cant_vectors es la cantidad de vectores que vamos a tomar en cuenta para achicar el data
  # o si cant_vectors es null, se usa una cantidad tal que cumpla que el porcentage
  # de variabilidad de los datos sea mayor igual a percentage
  # Se recorre el data para extraer como array los valores de los distintos atributos numéricos
  data = copy.deepcopy(data)
  numeric_attributes_values = {}
  data_len = len(data)
  numeric_atts_len = len(numeric_atts)
  for instance in data:
    for attribute in numeric_atts:
      #  Si se encuentra un valor numérico se agrega al diccionario
      if isinstance(instance[attribute], np.float64) or isinstance(instance[attribute], float) or isinstance(instance[attribute], int):
        if attribute in numeric_attributes_values:
          numeric_attributes_values[attribute].append(instance[attribute])
        else:
          numeric_attributes_values[attribute] = [instance[attribute]]
  
  original_means = []
  for att, values in numeric_attributes_values.items():
    values_np = np.asarray(values)
    mean = values_np.mean()
    original_means.append(mean)
    i = -1
    for val in values:
      i += 1
      data[i][att] -= mean

  # matriz de covarianza 
  #     FILA 0        FILA 1
  # [[col0, col1], [col0, col1]]
  covariance_matrix = get_covariance_matrix(data, numeric_atts, numeric_atts_len, data_len)
  # print('covariance matrix')
  # print(covariance_matrix)

  eigen_vectors, eigen_values, v = np.linalg.svd(covariance_matrix)

  row_data = transpose_and_format_data(data, numeric_atts_len)
  row_eigen_vectors = transpose_and_format_data(eigen_vectors, numeric_atts_len)

  
  if (cant_vectors):
    # Me quedo con cant_vectors cantidad de vectores que 
    # determinan mayor variabilidad de los datos
    row_eigen_vectors = row_eigen_vectors[:cant_vectors]
    sum_percentage = 0
    total = sum(eigen_values)
    for value in eigen_values[:cant_vectors]:
      sum_percentage += value * 100 / total
  elif percentage:
    # Me quedo con x cantidad de vectores que determinen más de 
    # percentage% de variabilidad de los datos
    cant_vectors, sum_percentage = get_cant_vectors_gte_percentage(eigen_values, percentage)
    row_eigen_vectors = row_eigen_vectors[:cant_vectors]
    
  print()
  print('La cantidad de componentes principales utilizadas es:', cant_vectors)
  print('de un total de:', numeric_atts_len, "atributos")
  print('que suman un', "%.2f" % sum_percentage, '% de variabilidad de los datos')

  matrix_T = multiply_matrix(row_eigen_vectors, row_data)
  matrix = undo_transpose(matrix_T)

  final_data = format_PC_data(matrix)
  
  return final_data, row_eigen_vectors, original_means

def format_PC_data(matrix):
  new_data = []
  atts_dict = {}
  for i in range(len(matrix[0])): 
    atts_dict['PC_' + str(i+1)] = None
  for row in matrix: 
    i = -1  
    new_data.append(copy.deepcopy(atts_dict))
    for value in row:
      i += 1
      new_data[-1]['PC_' + str(i+1)] = row[i]
  return new_data

def get_original_data(row_eigen_vectors, matrix, original_means):
  row_eigen_vectors_T = undo_transpose(row_eigen_vectors)
  original_data = multiply_matrix(row_eigen_vectors_T, matrix)
  i = -1
  for instance in original_data:
    i += 1
    j = -1
    for value in instance:
      j += 1
      original_data[i][j] += original_means[i]

  return undo_transpose(original_data)

def get_cant_vectors_gte_percentage(eigen_values, percentage):
  percentage_sum = 0
  total = sum(eigen_values)
  i = 0
  for value in eigen_values:
    i += 1
    percentage_sum += value * 100 / total
    if percentage_sum >= percentage:
      return i, percentage_sum
  return i, percentage_sum

def get_covariance_matrix(data, attributes, atts_len, data_len):
  # Devuelve un np array que representa la matriz de covarianza
  matrix = []
  for k in range(atts_len):
    row = []
    for i in range(data_len):
      for j in range(atts_len):
        if len(row) < j+1:
          row.append(data[i][attributes[k]] * data[i][attributes[j]])
        else:
          row[j] += data[i][attributes[k]] * data[i][attributes[j]]

    for index in range(len(row)):
      row[index] /= (data_len - 1)
    matrix.append(row)

  return np.array(matrix)


def transpose_and_format_data(data, atts_len):
  ret = []
  for i in range(atts_len): ret.append([])
  for instance in data:
    if isinstance(instance, dict): instance = list(instance.values())
    for i in range(atts_len):
      ret[i].append(instance[i])
  return ret


def undo_transpose(data):
  ret = []
  for i in range(len(data[0])): ret.append([])
  for i in range(len(ret)):
    for instance in data:
      ret[i].append(instance[i])
  return ret


def multiply_matrix(a,b):
    zip_b = zip(*b)
    zip_b = list(zip_b)
    return [[sum(ele_a*ele_b for ele_a, ele_b in zip(row_a, col_b)) 
             for col_b in zip_b] for row_a in a]


def one_hot_encoding(data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes):
  # Data tiene que venir sin el target attribute 
  # y como array de arrays, donde cada array es una fila
  att_values  = {}
  for x in categorical_atts: att_values[x] = []
  for instance in data:
    i = -1
    for index in categorical_atts_indexes:
      i += 1
      att = categorical_atts[i]
      att_values[att].append([instance[index]])

  onehot_encoded_data = one_hot_encoder(categorical_atts, att_values, non_categorical_atts, non_categorical_atts_indexes, data)

  return onehot_encoded_data

  # ABAJO ES USANDO SKLEARN, PERO NO ME SIRVIO
  # label_encoder = LabelEncoder()
  # integer_encoded_data = {}
  # for x in categorical_atts: integer_encoded_data[x] = None
  # for att, values in att_values.items():
  #   integer_encoded = label_encoder.fit_transform(values)
  #   integer_encoded_data[att] = integer_encoded

  # integer encode
  # integer_encoded_data = label_encoder(categorical_atts, att_values)
  # print(integer_encoded_data)


  # binary encode
  # onehot_encoder = OneHotEncoder(sparse=False)
  # onehot_encoded_data = {}
  # for x in categorical_atts: onehot_encoded_data[x] = None
  # for att, integer_encoded in integer_encoded_data.items():
  #   integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
  #   onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
  #   onehot_encoded_data[att] = onehot_encoded.tolist()

  
  # for x in onehot_encoded_data: print(); print(x)

  # new_data = []
  # attributes_dict = {}
  # for att in non_categorical_atts:
  #   attributes_dict[att] = None
  # for att, encoded in onehot_encoded_data.items():
  #   for i in range(len(encoded[0])):
  #     attributes_dict[att + '_' + str(i)] = None

  # i = -1
  # for instance in data:
  #   i += 1
  #   new_data.append(copy.deepcopy(attributes_dict))
  #   att_index = -1
  #   for att in non_categorical_atts:
  #     att_index += 1
  #     new_data[-1][att] = instance[non_categorical_atts_indexes[att_index]]
  #   for att, encoded in onehot_encoded_data.items():
  #     for j in range(len(encoded[i])):
  #       new_data[-1][att + '_' + str(j)] = encoded[i][j]
  


def one_hot_encoder(categorical_atts, att_values, non_categorical_atts, non_categorical_atts_indexes, data):
  integer_encoded_data = label_encoder(categorical_atts, att_values)

  onehot_encoded_data_dict = {'A1_Score_0': 0.0,'A1_Score_1': 0.0,'A2_Score_0': 0.0,'A2_Score_1': 0.0,'A3_Score_0': 0.0,'A3_Score_1': 0.0,'A4_Score_0': 0.0,'A4_Score_1': 0.0,'A5_Score_0': 0.0,'A5_Score_1': 0.0,'A6_Score_0': 0.0,'A6_Score_1': 0.0,'A7_Score_0': 0.0,'A7_Score_1': 0.0,'A8_Score_0': 0.0,'A8_Score_1': 0.0,'A9_Score_0': 0.0,'A9_Score_1': 0.0,'A10_Score_0': 0.0,'A10_Score_1': 0.0,
                              'gender_0': 0.0, 'gender_1': 0.0, 'jundice_0': 0.0, 'jundice_1': 0.0, 'austim_0': 0.0, 'austim_1': 0.0, 'used_app_before_0': 0.0, 'used_app_before_1': 0.0, 'age_desc_0': 0.0}
  onehot_encoded_data = []

  ethnicity_possible_values = ['White-European','Latino','Others','Black','Asian',"'Middle Eastern '",'Pasifika',"'South Asian'",'Hispanic','Turkish','others']
  for i in range(len(ethnicity_possible_values)): onehot_encoded_data_dict['ethnicity_' + str(i)] = 0.0
  relation_possible_values = ['Self','Parent',"'Health care professional'",'Relative','Others']
  for i in range(len(relation_possible_values)): onehot_encoded_data_dict['relation_' + str(i)] = 0.0
  country_possible_values = ["'United States'","Brazil","Spain","Egypt","'New Zealand'","Bahamas","Burundi","Austria","Argentina","Jordan","Ireland","'United Arab Emirates'","Afghanistan","Lebanon","'United Kingdom'","'South Africa'","Italy","Pakistan","Bangladesh","Chile","France","China","Australia","Canada","'Saudi Arabia'","Netherlands","Romania","Sweden","Tonga","Oman","India","Philippines","'Sri Lanka'","'Sierra Leone'","Ethiopia","'Viet Nam'","Iran","'Costa Rica'","Germany","Mexico","Russia","Armenia","Iceland","Nicaragua","'Hong Kong'","Japan","Ukraine","Kazakhstan","AmericanSamoa","Uruguay","Serbia","Portugal","Malaysia","Ecuador","Niger","Belgium","Bolivia","Aruba","Finland","Turkey","Nepal","Indonesia","Angola","Azerbaijan","Iraq","'Czech Republic'",'Cyprus']
  for i in range(len(country_possible_values)): onehot_encoded_data_dict['contry_of_res_' + str(i)] = 0.0

  values = list(integer_encoded_data.values())
  for i in range(len(values[0])):
    onehot_encoded_data.append(copy.deepcopy(onehot_encoded_data_dict))
    j = -1
    for att in non_categorical_atts:
      j += 1
      onehot_encoded_data[-1][att] = data[i][non_categorical_atts_indexes[j]]

  for att, values in integer_encoded_data.items():
    i = -1
    for value in values:
      i += 1
      onehot_encoded_data[i][att + '_' + str(value)] = 1.0

  
  
  return onehot_encoded_data

def label_encoder(categorical_atts, att_values):
  integer_encoded_data = {}
  for x in categorical_atts: integer_encoded_data[x] = None
  for att, values in att_values.items():
    integer_encoded = []
    for value in values:
      if (att == 'A1_Score' or att == 'A2_Score' or att == 'A3_Score' or att == 'A4_Score' or att == 'A5_Score' or att == 'A6_Score' or att == 'A7_Score' or att == 'A8_Score' or att == 'A9_Score' or att == 'A10_Score'):
        int_values = [0,1]
        possible_values = ['0','1']
        integer_encoded.append(int_values[possible_values.index(value[0])])
      elif (att == 'gender'):
        int_values = [0,1]
        possible_values = ['f','m']
        integer_encoded.append(int_values[possible_values.index(value[0])])
      elif (att == 'ethnicity'):
        int_values = [0,1,2,3,4,5,6,7,8,9,10]
        possible_values = ['White-European','Latino','Others','Black','Asian',"'Middle Eastern '",'Pasifika',"'South Asian'",'Hispanic','Turkish','others']
        integer_encoded.append(int_values[possible_values.index(value[0])])
      elif (att == 'jundice' or att == 'austim' or att == 'used_app_before'):
        int_values = [0,1]
        possible_values = ['no','yes']
        integer_encoded.append(int_values[possible_values.index(value[0])])
      elif (att == 'contry_of_res'):
        possible_values = ["'United States'","Brazil","Spain","Egypt","'New Zealand'","Bahamas","Burundi","Austria","Argentina","Jordan","Ireland","'United Arab Emirates'","Afghanistan","Lebanon","'United Kingdom'","'South Africa'","Italy","Pakistan","Bangladesh","Chile","France","China","Australia","Canada","'Saudi Arabia'","Netherlands","Romania","Sweden","Tonga","Oman","India","Philippines","'Sri Lanka'","'Sierra Leone'","Ethiopia","'Viet Nam'","Iran","'Costa Rica'","Germany","Mexico","Russia","Armenia","Iceland","Nicaragua","'Hong Kong'","Japan","Ukraine","Kazakhstan","AmericanSamoa","Uruguay","Serbia","Portugal","Malaysia","Ecuador","Niger","Belgium","Bolivia","Aruba","Finland","Turkey","Nepal","Indonesia","Angola","Azerbaijan","Iraq","'Czech Republic'",'Cyprus']
        int_values = [i for i in range(len(possible_values))]
        integer_encoded.append(int_values[possible_values.index(value[0])])
      elif (att == 'relation'):
        possible_values = ['Self','Parent',"'Health care professional'",'Relative','Others']
        int_values = [i for i in range(len(possible_values))]
        integer_encoded.append(int_values[possible_values.index(value[0])])
      elif (att == 'age_desc'):
        integer_encoded.append(0)

      integer_encoded_data[att] = array(integer_encoded)
  return integer_encoded_data

def decode_data (data):
  # Devuelve un arreglo de arreglos donde cada elemento es una fila
  result = []
  for i in range(len(data)):
    result.append([])
    for j in range(len(data[i])):
      if(isinstance(data[i][j], np.bytes_)):
        result[i].append(data[i][j].decode())
      else:
        result[i].append(data[i][j])
  return result

def extract_target_attributes(data):
  target_attributes = []
  i = -1
  for instance in data:
    i += 1
    target_attributes.append(instance[-1])
    data[i].pop()
  return (data, target_attributes)

def insert_target_attributes(data, target_attr, target_attributes):
  for i in range(len(data)):
    data[i][target_attr] = target_attributes[i]

def PCA_validation_and_training_data(numeric_training_set, numeric_validation_set, training_target_attributes, validation_target_attributes, attributes, target_attr, PCA_cant_vectors, PCA_percentage):
  PC_training_data, row_eigen_vectors, original_means = PCA(numeric_training_set, attributes, attributes, PCA_cant_vectors, PCA_percentage)

  PC_attributes = list(PC_training_data[0].keys())

  # METO EL TARGET ATTRIBUTE
  insert_target_attributes(PC_training_data, target_attr, training_target_attributes)

  # PC_training_data TIENE LA DATA DESP DE HACER PCA
  # print(PC_training_data)

  # RESTO LA MEDIA A CADA ATRIBUTO
  row_data = transpose_and_format_data(numeric_validation_set, len(attributes))
  for i in range(len(row_data)):
    for j in range (len(row_data[0])):
      row_data[i][j] -= original_means[i]

  # print()
  # print(len(row_data[0]))

  matrix_T = multiply_matrix(row_eigen_vectors, row_data)
  matrix = undo_transpose(matrix_T)


  PC_validation_data = format_PC_data(matrix)

  # METO EL TARGET ATTRIBUTE
  insert_target_attributes(PC_validation_data, target_attr, validation_target_attributes)
  
  return PC_training_data, PC_validation_data, PC_attributes

def format_data_chi(attributes, data):
  chi_atts_dict = {}
  new_data = []
  for i in range(len(attributes)): 
    chi_atts_dict['CHI_' + str(i+1)] = None

  for instance in data:
    new_data.append(copy.deepcopy(chi_atts_dict))
    for i in range(len(attributes)):
      new_data[-1]['CHI_' + str(i+1)] = instance[attributes[i]]

  return new_data
  

def get_n_max_indexes(data, n):
  max_indexes = []
  for i in range(n):
    index = data.index(max(data))
    max_indexes.append(index)
    data.pop(index)
  return max_indexes
  

if __name__ == "__main__":
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

  categorical_atts_indexes = [0,1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18]
  non_categorical_atts_indexes = [10]

  data = utils.process_missing_values(data_set, attributes, True)
  data = decode_data(data)

  # Se divide el conjunto de datos
  data_20, data_80 = utils.split_20_80(data)

  # Se arman los training y validation set 
  training_data, training_target_attributes = extract_target_attributes(data_80)

  numeric_training_set = one_hot_encoding(training_data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)

  validation_data, validation_target_attributes = extract_target_attributes(data_20)
  numeric_validation_set = one_hot_encoding(validation_data, categorical_atts, categorical_atts_indexes, non_categorical_atts, non_categorical_atts_indexes)

  numeric_attributes = list(numeric_training_set[0].keys())
  numeric_atts_len = len(numeric_attributes)
  print('cantidad de nuevos atributos', numeric_atts_len)
  


  #######################################################################################
  ###########################             PCA            ################################
  #######################################################################################
  PC_training_data, PC_validation_data, PC_attributes = PCA_validation_and_training_data(copy.deepcopy(numeric_training_set), copy.deepcopy(numeric_validation_set), 
                                                        copy.deepcopy(training_target_attributes), copy.deepcopy(validation_target_attributes), 
                                                        numeric_attributes, target_attr, i+1, 95)

                                                                              # (data, validation_set, target_attr, attributes, k, weight, normalize, use_standarization):
  errors_KNN = KNN.holdout_validation(PC_training_data, PC_validation_data, target_attr, PC_attributes, 3, True, True, True)
  print()
  print('cantidad de errores KNN/PCA:',errors_KNN)

  nb_classifier = NaiveBayes(PC_training_data, PC_attributes, target_attr)
  errors_NB = nb_classifier.holdout_validation(PC_validation_data, target_attr)
  print('cantidad de errores NB/PCA:',errors_NB)


  #######################################################################################
  ###########################             CHI2           ################################
  #######################################################################################

  training_set_array = []
  for x in copy.deepcopy(numeric_training_set):
    training_set_array.append(list(x.values()))
  
  chi2_results = chi2(training_set_array, training_target_attributes)
  chi2_result_list = chi2_results[0].tolist()

  max_indexes = get_n_max_indexes(chi2_result_list, i+1)
  max_attributes = []
  for j in max_indexes:
    max_attributes.append(numeric_attributes[j])

  chi_training_set = format_data_chi(max_attributes, copy.deepcopy(numeric_training_set))
  chi_attributes = list(chi_training_set[0].keys())

  insert_target_attributes(chi_training_set, target_attr, training_target_attributes)

  chi_validation_set = format_data_chi(max_attributes, copy.deepcopy(numeric_validation_set))
  insert_target_attributes(chi_validation_set, target_attr, validation_target_attributes)

  chi_errors_KNN = KNN.holdout_validation(chi_training_set, chi_validation_set, 
                                          target_attr, chi_attributes, 3, True, True, True)
  print()
  print('cantidad de errores KNN/CHI2:',chi_errors_KNN)

  nb_classifier = NaiveBayes(chi_training_set, chi_attributes, target_attr)
  chi_errors_NB = nb_classifier.holdout_validation(chi_validation_set, target_attr)
  print('cantidad de errores NB/CHI2:',chi_errors_NB)
