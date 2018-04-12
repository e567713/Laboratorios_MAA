import utils

#########################
# Ejercicio del teórico #
#########################

# Data set del teórico
S = [
    {'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
    {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Malo', 'Salva': 'Yes'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Bueno', 'Salva': 'No'},
]

# Data set de prueba
S2 = [{'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Matutino',
       'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
      {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
       'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
      {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
       'Humedad': 'Media', 'Humor Docente': 'Malo', 'Salva': 'Yes'},
      {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Matutino',
       'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'No'}]

# Data set de prueba valores faltantes
S3 = [
    {'Dedicacion': 'Alta', 'Dificultad': 'Alta', 'Horario': 'Matutino',
        'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
    {'Dedicacion': 'Baja', 'Dificultad': 'Media', 'Horario': 'Matutino',
        'Humedad': 'Alta', 'Humor Docente': 'Malo', 'Salva': 'No'},
    {'Dedicacion': 'Media', 'Dificultad': 'Alta', 'Horario': 'Nocturno',
        'Humedad': 'Media', 'Humor Docente': 'Malo', 'Salva': 'Yes'},
    {'Dedicacion': 'Media', 'Dificultad': '?', 'Horario': 'Matutino',
        'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'No'},
    {'Dedicacion': 'Baja', 'Dificultad': '?', 'Horario': 'Matutino',
        'Humedad': 'Media', 'Humor Docente': 'Bueno', 'Salva': 'Yes'},
]


print('-------------------------------------------------')
print('-------------     Ejercicio 5a     --------------')
print('-------------------------------------------------')
print('')
print('Aplicacion de ID3 al ejemplo visto en el teórico')
print('')

# S_entropy = utils.entropy(S, 'Salva')
# print('Entropía del conjunto: ', S_entropy)

# S_information_gain = utils.information_gain(S, 'Dedicacion', 'Salva', False)
# print('Information gain del atributo Dedicación: ', S_information_gain)
# S_information_gain = utils.information_gain(S, 'Humor Docente', 'Salva', False)
# print('Information gain del atributo Humor Docente: ', S_information_gain)
# S_information_gain = utils.information_gain(S, 'Horario', 'Salva', False)
# print('Information gain del atributo Horario: ', S_information_gain)
# S_information_gain = utils.information_gain(S, 'Dificultad', 'Salva', False)
# print('Information gain del atributo Dificultad: ', S_information_gain)
# S_information_gain = utils.information_gain(S, 'Humedad', 'Salva', False)
# print('Information gain del atributo Humedad: ', S_information_gain)

tree = utils.ID3_algorithm(
    S,
    ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
    'Salva',
    True, False)

utils.print_tree(tree, tree['data'], None, True, '')

print()
print()
print('Aplicacion de ID3 a un segundo conjunto de entrenamiento')
print()

# Algoritmo aplicado al segundo conjunto de prueba
tree2 = utils.ID3_algorithm(
    S2,
    ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
    'Salva',
    True, False)

utils.print_tree(tree2, tree['data'], None, True, '')


#############################################
# Ejercicio con el data set del laboratorio #
#############################################

# Leemos data set del laboratorio
examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos

print('')
print('')
print('')
print('-------------------------------------------------')
print('-------------     Ejercicio 5b     --------------')
print('-------------------------------------------------')
print('')

# Calculamos su entropía.
data_set_entropy = utils.entropy(data_set, 'Class/ASD')
print('Entropía del conjunto: ', data_set_entropy)
print()

print('Aplicacion de ID3 extendido para manejar atributos numéricos')
print('')


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

# Primera solución implementada
print('Utilizando thresholds')
print()
tree_2 = utils.ID3_algorithm_with_threshold(
    data_set,
    attributes,
    'Class/ASD',
    ['age'], False)

utils.print_tree(tree_2, tree_2['data'], None, True, '')

print('Utilizando gain ratio')
print()
# Segunda solución implementada
tree_3 = utils.ID3_algorithm(
    data_set,
    attributes,
    'Class/ASD',
    False, False)
utils.print_tree(tree_3, tree_3['data'], None, True, '')

print()
print('Aplicacion de ID3 extendido para manejar valores faltantes')
print()
print('Más común')
print('')

# Primera solución implementada valores faltantes
tree_4 = utils.ID3_algorithm(
    S3,
    ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
    'Salva',
    False, False)
utils.print_tree(tree_4, tree_4['data'], None, True, '')

print()
print()
print('Probabilidades')
print('')
# Segunda solución implementada valores faltantes
tree_5 = utils.ID3_algorithm(
    S3,
    ['Dedicacion', 'Dificultad', 'Horario', 'Humedad', 'Humor Docente'],
    'Salva',
    False, True)
utils.print_tree(tree_5, tree_5['data'], None, True, '')

print('')
print('')
print('')
print('-------------------------------------------------')
print('-------------     Ejercicio 5c     --------------')
print('-------------------------------------------------')
print('')

# Calculamos su entropía.
data_set_entropy = utils.entropy(data_set, 'Class/ASD')
print('Entropía del conjunto: ', data_set_entropy)


# Separamos el data set en dos subconjuntos
print()
print('Se separa el data set en dos subconjuntos')
splitted_data = utils.split_20_80(data_set)

# Verificamos la correctitud de los tamaños
print('Tamaño del data set original: ', str(len(data_set)))
print('Tamaño del subset de validación: ', str(len(splitted_data[0])))
print('Tamaño del subset de entrenamiento: ', str(len(splitted_data[1])))
print()
	
# Parte 1
print('Parte 1')
# Se realiza cross-validation de tamaño 10 sobre el 80% del conjunto original.
print('Se realiza 10-fold cross-validation')

v_cs = utils.cross_validation(
    splitted_data[1], attributes, 'Class/ASD', 10)
print('Promedio de error: ', v_cs)
# Parte 2
print('Parte 2')
print('Se realiza Hold out validation')

# Se entrena con el 80%
tree_6 = utils.ID3_algorithm(
    splitted_data[1],
    attributes,
    'Class/ASD',
    False, False)

# Se valida con el 20%
v_ho = utils.validation(
    tree_6, splitted_data[0], 'Class/ASD')
print('Resultado de la validación: ', v_ho)