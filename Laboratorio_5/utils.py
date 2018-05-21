import numpy as np
import scipy
import copy
import utils
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import chi2

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
