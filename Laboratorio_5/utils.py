from collections import Counter

import numpy as np
import scipy
import copy

from scipy.io import arff

import utils
from numpy import array
import math
import KNN

def read_file(path):
    return arff.loadarff(path)

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


def process_missing_values(data, attributes, use_most_common):
    # Procesa el conjunto de datos para atacar el problema de valores incompletos.

    # Guardará los valores más comunes para los atributos que necesiten calcularse.
    most_common = {}
    i = len(data)

    # Se itera de atrás para adelante para poder borrar por indice si not use_most_common
    for instance in reversed(data):
        i -= 1
        for attribute in attributes:

            # Si se encuentra un valor faltante.
            if ((isinstance(instance[attribute], np.float64) and np.isnan(instance[attribute])) or (isinstance(instance[attribute], np.bytes_) and instance[attribute] == b'?')):
                # Si use_most_common, se cambia el valor faltante por el más común del atributo,
                # si no, se descarta el ejemplo
                if use_most_common:
                    # Si no se ha calculado el valor más común, se lo calcula y almacena.
                    if not attribute in most_common:
                        most_common[attribute]=found_most_common_attribute_value(
                            data, attribute)

                    # Se cambia el valor faltante por el más común.
                    instance[attribute] = most_common[attribute]
                else:
                    # Se descarta el ejemplo
                    data = np.delete(data,i)
                    break
    return data

def found_most_common_attribute_value(data, attribute):
    values = [instance[attribute] for instance in data if not (isinstance(instance[attribute], np.float64) and np.isnan(instance[attribute])) or (isinstance(instance[attribute], np.bytes_) and instance[attribute] == b'?')]
    data = Counter(values)
    return max(values, key=data.get)

def split_20_80(d):
    # Divide el conjunto de datos en dos subconjuntos, uno con el 80% de los datos
    # y el otro con el restante 20%.

    # Se copia el conjunto de datos original para no alterarlo.
    data = copy.deepcopy(d)

    # Se ordenan aleatoriamente los ejemplos del conjunto para simular la
    # elección al azar de elementos para formar los subconjuntos.
    np.random.shuffle(data)

    # Se obtiene la cantidad de elementos que suponen el 20% de los datos.
    limit = len(data) // 5

    # Se crean los subconjuntos
    subset_20 = data[:limit]
    subset_80 = data[limit:]

    return (subset_20, subset_80)

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
            elif (att == 'country_of_res'):
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

def insert_sesgo_one(data):
    for i in range(len(data)):
        data[i]["sesgo"] = 1


# def scalarProduct(weight, instance):
#     result=0
#     if (len(weight)== len(instance)):
#         result = weight[0]
#         for i in range(len(weight)-1):
#             result+= weight[i+1]*instance[i]
#     return result

def scale(data, attributes, use_standarization):
    # 'data' es es el conjunto de entrenamiento
    # 'attributes' un array con los nombres de los atributos
    # si 'use_standarization' = true, normaliza usando media y varianza
    # si no, usa min-max
    # Se retorna (data, scalation_params) siendo el último {att: (mean, std)}
    # en caso de use_stand = true, o {att: (min, max)} en caso de use_stand = false
    numeric_attributes_values = {}
    scalation_parameters = {}
    # Se recorre el data para extraer los valores de los distintos atributos numéricos
    for instance in data:
        for attribute in attributes:
            #  Si se encuentra un valor numérico se agrega al diccionario
            if isinstance(instance[attribute], np.float64) or isinstance(instance[attribute], float) or isinstance(instance[attribute], int):
                if attribute in numeric_attributes_values:
                    numeric_attributes_values[attribute].append(instance[attribute])
                else:
                    numeric_attributes_values[attribute] = [instance[attribute]]
    # print("//////////////")
    # print(numeric_attributes_values)
    # print("//////////////")
    for att, values in numeric_attributes_values.items():
        values_np = np.asarray(values)
        if use_standarization:
            print('TODO')
            return
            # scaled = (values_np - values_np.mean()) / values_np.std()
            # scalation_parameters[att] = (values_np.mean(), values_np.std())
            # i = -1
            # for instance in data:
            #     i += 1
            #     instance[att] = scaled[i]
        else:
            if (min(values) == max(values)):
                minmax = [0.0 for values_i in values]
            else:
                minmax = [(values_i - min(values)) / (max(values) - min(values)) for values_i in values]
                scalation_parameters[att] = (min(values), max(values))
                i = -1
                for instance in data:
                    i += 1
                    instance[att] = minmax[i]
    return(data, scalation_parameters)

def scalarProductDict(weight, instance, attributesWithSesgo):
    result=0
    weightIndex=0
    for attr in attributesWithSesgo:
        result += weight[weightIndex]*instance[attr]
        weightIndex+=1
    # print("scalar")
    # print(result)
    return result

def calculateH0(weight, instance, attributesWithSesgo):
    oTx= scalarProductDict(weight, instance, attributesWithSesgo)
    eExp = (math.e)**(-oTx)
    h0 = 1/(1+(eExp))
    return h0

def instanceCost(weight, instance, attributesWithSesgo, target_attr):
    h0 = calculateH0(weight, instance, attributesWithSesgo)
    if (instance[target_attr]== 'YES'):
        # print("Yes")
        # print(h0)
        # print(-math.log10(h0))
        return -math.log10(h0)
    elif (instance[target_attr]== 'NO'):
        # print("No")
        # print(h0)
        # print(-math.log10(1-h0))
        return -math.log10(1-h0)

def costFunction(weight, data, attributesWithSesgo, target_attr):
    lenght = len(data)
    cost=0
    for instance in data:
        cost += instanceCost(weight,instance,attributesWithSesgo, target_attr)
    return cost/lenght

def descentByGradient(weight, data, a, attributesWithSesgo, target_attr):
    newWeight = {}
    weightLenght = len(weight)
    dataLenght = len(data)
    for j in range(weightLenght):
        sum = 0
        for instance in data:
            ih0 = calculateH0(weight, instance, attributesWithSesgo)
            if instance[target_attr] == 'YES':
                sum += (ih0-1)*instance[attributesWithSesgo[j]]
            else:
                sum += ih0*instance[attributesWithSesgo[j]]
        newWeight[j] = weight[j] - ((a*sum)/dataLenght)
    return newWeight

def clssify_LR_instance(instance, weight, attributesWithSesgo):
    result = 'YES' if (calculateH0(weight, instance, attributesWithSesgo) > 0.5) else "NO"
    return result

def KNN_holdout_validation(data, validation_set, target_attr, attributes, k, weight):
    # retorna (len(validation_set), cantidad de errores, promedio de errores)
    errors = 0
    for instance in validation_set:
        instance_copy = copy.deepcopy(instance)
        result = KNN.classify(instance_copy, data, k, target_attr, weight, attributes)
        if result != instance[target_attr]:
            errors += 1
    return (len(validation_set), errors, errors/len(validation_set))


def scale_instance(instance, scalation_parameters, use_standarization):
    for att, parameters in scalation_parameters.items():
        mean_or_min = parameters[0]
        std_or_max = parameters[1]
        if use_standarization:
            instance[att] = ((instance[att] - mean_or_min) / std_or_max)
        else:
            instance[att] = (instance[att] - mean_or_min) / (std_or_max - mean_or_min)
    return instance


def LR_holdout_validation(validation_set, target_attr, weight, LR_numeric_attributes):
    errors = 0
    for instance in validation_set:
        result = clssify_LR_instance(instance, weight, LR_numeric_attributes)
        if (instance[target_attr]!=result):
            errors += 1
    return errors