import math
from scipy.io import arff
from collections import Counter
import copy
import random
import numpy as np


def ID3_algorithm(data, attributes, target_attr, use_information_gain, use_missing_values_first_method):
    # Algoritmo ID3 básico.

    # Genera lista únicamente con los valores del target attribute.
    target_attr_values = [instance[target_attr] for instance in data]

    # Si todas las instancias tienen el mismo valor → etiquetar con ese valor.
    if (target_attr_values[1:] == target_attr_values[:-1]):
        return {
            'data': target_attr_values[0],
            'childs': {}
        }

    # Si no me quedan atributos → etiquetar con el valor más común.
    elif not attributes:
        return {
            'data': most_common(target_attr_values),
            'childs': {}
        }

    # En caso contrario
    else:
        # Se obtiene el atributo best_attr que mejor clasifica los ejemplos.
        best_attr = get_best_attribute(
            data, attributes, target_attr, use_information_gain, use_missing_values_first_method)

        # Se obtienen los valores que puede tomar el atributo best_attr.
        best_attr_values = set(instance[best_attr] for instance in data)

        # Se genera un árbol con el atributo best_attr en su raíz.
        tree = {
            'data': best_attr,
            'childs': {}
        }
        for value in best_attr_values:

            # Nos quedamos con los ejemplos de data que tengan el valor
            # value para el atributo best_attribute.
            filtered_data = [
                instance for instance in data if instance[best_attr] == value]

            # Si filtered_data es vacío → etiquetar con el valor más probable.
            if not filtered_data:
                tree['childs'][value] = {
                    'data': most_common(target_attr_values),
                    'childs': {}
                }

            # Si filtered_data no es vacío
            else:
                # Se quita a best_attr de la lista de atributos.
                filtered_attributes = copy.deepcopy(attributes)
                filtered_attributes.remove(best_attr)
                tree['childs'][value] = ID3_algorithm(
                    filtered_data, filtered_attributes, target_attr, use_information_gain, use_missing_values_first_method)
        return tree


def entropy(data, target_attr):
    # Calcula la entropía del conjunto data dado para el atributo target_attr.

    frequencies = {}
    data_entropy = 0.0

    # Calcula la frecuencia de cada valor en el atributo objetivo.
    for instance in data:
        if (instance[target_attr] in frequencies):
            frequencies[instance[target_attr]] += 1.0
        else:
            frequencies[instance[target_attr]] = 1.0

    # Para cada valor del atributo objetivo se calcula su proporción
    # dentro del conjunto data y se aplica la fórmula de entropía.
    for frequency in frequencies.values():
        data_entropy -= (frequency / len(data)) * \
            math.log(frequency / len(data), 2)

    return data_entropy


def information_gain(data, attr, target_attr, use_missing_values_first_method):
    # Calcula la ganancia de información del atributo attr sobre el
    # conjunto data.

    data_subsets = {}
    data_information_gain = 0.0

    # Se divide el conjunto data en subconjuntos que tienen en común
    # el valor del atributo attr.
    for instance in data:
        if (instance[attr] in data_subsets):
            data_subsets[instance[attr]].append(instance)
        else:
            data_subsets[instance[attr]] = [instance]

    # Extension para valores faltantes primer metodo
    if '?' in data_subsets:
        if use_missing_values_first_method:
            common_value = find_most_common_value_in_S(data_subsets, target_attr)
            data_subsets[common_value].extend(data_subsets['?'])
            data_subsets.pop('?', None)
        else:
            return information_gain_missing_values_second_method(data, attr, target_attr)

    # Se calcula el valor de information gain según lo visto en teórico.
    data_information_gain = entropy(data, target_attr)
    for data_subset in data_subsets.values():
        data_information_gain -= (len(data_subset) / len(data)) * \
            entropy(data_subset, target_attr)
    return data_information_gain


def read_file(path):
    return arff.loadarff(path)


def most_common(lst):
    # Retorna el elemento más común dentro de la lista pasada por parámetro.
    #   https://stackoverflow.com/questions/1518522/python-most-common-element-in-a-list
    data = Counter(lst)
    return max(lst, key=data.get)


def get_best_attribute(data, attributes, target_attr, use_information_gain, use_missing_values_first_method):
    # Elige el mejor atributo medido según la ganancia de información o según
    # el gain ratio dependiendo del valor del parametro booleano use_information_gain.
    # Si existe más de un atributo óptimo para las condiciones dadas,
    # se devuelve uno aleatorio entre ellos.
    maximum_values_tied = []
    max_measure = -1
    for attr in attributes:
        if use_information_gain:
            measure = information_gain(data, attr, target_attr, use_missing_values_first_method)
        else:
            measure = gain_ratio(data, attr, target_attr, use_missing_values_first_method)
        if measure > max_measure:
            max_measure = measure
            maximum_values_tied = []
            maximum_values_tied.append(attr)
        elif measure == max_measure:
            maximum_values_tied.append(attr)
    return random.choice(maximum_values_tied)


def print_tree(tree, attr, childIsSheet, first, tab):
    if type(attr).__name__ == 'bytes_':
        print(tab , str(attr, 'utf-8'))
    else:    
        print(tab , attr)
    if not childIsSheet and not first:
        tab = '   ' + tab[:len(tab) - 3] + '  |--'
        if type(tree['data']).__name__ == 'bytes_':
            print(tab , str(tree['data'], 'utf-8'))
        else:
            print(tab , tree['data'])
    if tree['childs']:
        for key, value in tree['childs'].items():
            if value['childs']:
                print_tree(value, key, False, False, '   ' +
                           tab[:len(tab) - 3] + '  |--')
            else:
                print_tree(value, key, True, False, '   ' +
                           tab[:len(tab) - 3] + '  |--')
    else:
        if type(tree['data']).__name__ == 'bytes_':
            print('   ' + tab[:len(tab) - 3] + '  |--' , str(tree['data'], 'utf-8'))
        else:
            print('   ' + tab[:len(tab) - 3] + '  |--' , tree['data'])


def find_most_common_value_in_S(instances, target_attr):
    # Instances es un diccionario con key un valor del attributo y value un arreglo con la fila que contiene valor key
    # EJ: {Alta: [{'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Nocturno', 'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'}]}
    # Busca el valor más comun que toma el atributo en todo S
    # Si existe más de un valor más comun, se devuelve uno aleatorio entre ellos.
    maximum_keys_tied = []
    max_quantity = -1
    for key, value in instances.items():
        if key != '?':
            if len(value) > max_quantity:
                max_quantity = len(value)
                maximum_keys_tied = []
                maximum_keys_tied.append(key)
            elif len(value) == max_quantity:
                maximum_keys_tied.append(key)
    return random.choice(maximum_keys_tied)


def ID3_algorithm_with_threshold(data, attributes, target_attr, numeric_attributes, use_missing_values_first_method):
    # Algoritmo ID3 extendido que divide el rango de valores posibles de los atributos
    # numéricos en dos. (Utiliza un único threshold para cada atributo)

    # Se cambian todos los valores de los atributos de tipo numeric
    splitted_data = split_numeric_attributes(
        copy.deepcopy(data), target_attr, numeric_attributes, use_missing_values_first_method)

    # Se llama a la función ID3 ya implementada con el conjunto de datos procesados.
    return ID3_algorithm(splitted_data, attributes, target_attr, True, False)


def split_numeric_attributes(data, target_attr, numeric_attributes, use_missing_values_first_method):
    # Recorre los atributos numéricos cambiando sus posibles valores según
    # un salto calclulado.

    for numeric_attr in numeric_attributes:
        # Se calcula el mejor threshold para el atributo numérico numeric_attr
        # midiendo según la gananacia de información.

        thresholds = []

        # Se ordenan los ejemplos en orden ascendente según los valores de numeric_attr.
        sorted_data = sorted(data, key=lambda x: x[numeric_attr])

        # Se recorre el conjunto data comparando de a 2 elementos para encontrar posibles
        # thresholds.
        for i in range(0, len(sorted_data) - 1):
            instance_1 = sorted_data[i]
            instance_2 = sorted_data[i + 1]

            # En caso de encontrar un posible candidato se almacena
            if instance_1[target_attr] != instance_2[target_attr] and instance_1[numeric_attr] != instance_2[numeric_attr]:
                thresholds.append(
                    (instance_1[numeric_attr] + instance_2[numeric_attr]) / 2)

        # Se recorre la lista de posibles thresholds.
        for threshold in thresholds:

            # Se dividen los valores de numeric_attr según el threshold dado.
            splitted_data = set_numeric_attribute_values(
                copy.deepcopy(data), numeric_attr, threshold)

            # Se busca el threshold que maximiza la ganancia de información.
            maximum_thresholds_tied = []
            max_ig = -1
            ig = information_gain(splitted_data, numeric_attr, target_attr, use_missing_values_first_method)
            if ig > max_ig:
                max_ig = ig
                maximum_thresholds_tied = []
                maximum_thresholds_tied.append(splitted_data)
            elif ig == max_ig:
                maximum_thresholds_tied.append(splitted_data)

        best_splitted_data = random.choice(maximum_thresholds_tied)

        # Se setean los valores del conjunto de datos según los resultados obtenidos
        # por el mejor threshold.
        for i in range(len(data)):
            data[i][numeric_attr] = best_splitted_data[i][numeric_attr]

    return data


def set_numeric_attribute_values(data, numeric_attr, threshold):
    # Divide los valores del atributo numérico numeric_attr según
    # el valor del threshold pasado por parámetro.

    new_key = numeric_attr + '>' + str(threshold)

    for instance in data:
        if instance[numeric_attr] > threshold:
            instance[numeric_attr] = 1
        else:
            instance[numeric_attr] = 0

    return data


def split_information(data, attr):
    # Función utilizada para el cálculo de la medida gain ratio.

    data_subsets = {}

    # Se divide el conjunto data en subconjuntos que tienen en común
    # el valor del atributo attr.
    for instance in data:
        if (instance[attr] in data_subsets):
            data_subsets[instance[attr]].append(instance)
        else:
            data_subsets[instance[attr]] = [instance]

    # Se calcula la entropía del conjunto data con respecto a los valores
    # del atributo attr (en lugar de hacerlo con respecto a los valores del
    # target attribute)
    attr_entropy = 1
    for data_subset in data_subsets.values():
        attr_entropy -= (len(data_subset) / len(data)) * \
            math.log(len(data_subset) / len(data), 2)

    return attr_entropy


def gain_ratio(data, attr, target_attr, use_missing_values_first_method):
    return  information_gain(data, attr, target_attr, use_missing_values_first_method) / split_information(data, attr)

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


def cross_validation(data, attributes, target_attr, k):
    # Implementación del algoritmo k-fold cross-validation
    # Nota: Recordar que el conjunto data fue seleccionado al azar del conjunto
    # inicial de datos.

    # Se divide el conjunto de datos en k subconjuntos.
    folds = np.array_split(data, k)

    # Lista para guardar los errores obtenidos en cada iteración del algoritmo.
    errors = []
    # Se abre el archivo para guardar los resultados

    for i in range(k):
        # Se aparta el subconjunto i para validación.
        validation_set = folds.pop(i)

        # Se unen los restantes subconjuntos para formar el nuevo set de entrenamiento.
        training_set = np.concatenate(folds)
        
        # Se entrena.
        tree = ID3_algorithm(training_set, attributes, target_attr, False, False)

        # Se verifica el resultado y se guarda el error cometido validado
        # con el subconjunto i.
        errors.append(validation(tree, validation_set, target_attr))

        # Se devuelve el subconjunto i a la lista de folds.
        folds.insert(i, validation_set)

    return sum(errors) / k


def validation(tree, validation_set, target_attr):
    errors = 0
    for instance in validation_set:
        errors += validate_instance(tree, instance, target_attr)
    return errors / len(validation_set)


def validate_instance(tree, instance, target_attr):
    if not tree['childs']:
        if (tree['data'] == instance[target_attr]):
            return 0
        else:
            return 1
    else:
        if instance[tree['data']] in tree['childs']:
            return validate_instance(tree['childs'][instance[tree['data']]], instance, target_attr)
        else:
            return 1


def information_gain_missing_values_second_method(data, attr, target_attr):
    # Calcula la ganancia de información del atributo attr sobre el
    # conjunto data.

    data_subsets = {}
    data_information_gain = 0.0
    quantity_subsets_target_attr = {}
    quantity_subsets = {}

    # Se divide el conjunto data en subconjuntos que tienen en común
    # el valor del atributo attr.
    # quantity_subsets es un diccionario con key un posible valor del atributo y value la cantidad de veces que aparece (|Sv|)
    # quantity_subsets_target_attr es un diccionario que tiene key el valor del atributo attr y value 
    # un array con tuplas (n,r) con n la cantidad de veces que aparece el resultado r en los ejemplos
    for instance in data:
        if (instance[attr] in data_subsets):
            data_subsets[instance[attr]].append(instance)

            if instance[attr] != '?':
                quantity_subsets[instance[attr]] += 1
                i = exist_tuple(quantity_subsets_target_attr[instance[attr]], instance[target_attr])
                if i == -1:
                    quantity_subsets_target_attr[instance[attr]].append((1, instance[target_attr]))
                else:
                    quantity_subsets_target_attr[instance[attr]][i] = (quantity_subsets_target_attr[instance[attr]][i][0] + 1, instance[target_attr])
        else:
            data_subsets[instance[attr]] = [instance]

            if instance[attr] != '?':
                quantity_subsets[instance[attr]] = 1
                quantity_subsets_target_attr[instance[attr]] = [(1, instance[target_attr])]
            
    # print()
    # print(attr)
    # print()
    # print(quantity_subsets)
    # print()
    # print(quantity_subsets_target_attr)
    
    if '?' in data_subsets:
        data_without_miss_value_instance = copy.deepcopy(data)
        for instance in data_subsets['?']:
            data_without_miss_value_instance.remove(instance)
        for key, value in quantity_subsets.items():
            quantity_subsets[key] += quantity_subsets[key] / len(data_without_miss_value_instance) * len(data_subsets['?'])
            for instance in data_subsets['?']:
                i = exist_tuple(quantity_subsets_target_attr[key], instance[target_attr])
                prop = len(data_subsets[key]) / len(data_without_miss_value_instance)
                if i == -1:
                    quantity_subsets_target_attr[key].append((prop, instance[target_attr]))
                else:
                    quantity_subsets_target_attr[key][i] = (quantity_subsets_target_attr[key][i][0] + prop, instance[target_attr])

    # print()
    # print(quantity_subsets)
    # print()
    # print(quantity_subsets_target_attr)

    # Entropy de todo el conjunto
    data_information_gain = entropy(data, target_attr)

    # Se calcula el valor de information gain según lo visto en teórico.
    for key, value in quantity_subsets.items():
        data_information_gain -= (value / len(data)) * \
            entropy_missing_values_second_method(key, quantity_subsets, quantity_subsets_target_attr)
    return data_information_gain


def exist_tuple(arr, target_value):
    i = -1
    for instance in arr:
        i += 1
        if instance[1] == target_value:
            return i
    return -1


def entropy_missing_values_second_method(key, quantity_subsets, quantity_subsets_target_attr):
    data_entropy = 0.0

    for element in quantity_subsets_target_attr[key]:
        data_entropy -= element[0] / quantity_subsets[key] * math.log(element[0] / quantity_subsets[key], 2)

    return data_entropy