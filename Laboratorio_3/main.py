import utils


# Leemos data set del laboratorio
examples = utils.read_file('Autism-Adult-Data.arff')
data_set = examples[0]  # Datos
metadata = examples[1]  # Metadatos


# Separamos el data set en dos subconjuntos
splitted_data = utils.split_20_80(data_set)

validation_set = splitted_data[0]
training_set = splitted_data[1]

utils.read_file

res = new Naive_Bayes(training_set)
res(instance) = clasificacion

nv.clasify(insance)

utils.v
