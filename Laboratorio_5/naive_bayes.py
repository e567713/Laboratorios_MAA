# coding=utf-8
import numpy as np
import math
from functools import reduce

class NaiveBayes:
    def __init__(self, data, attributes, target_attr):

        self.target_values_frecuency = {}
        self.target_attr = target_attr
        self.attributes = attributes
        self.total = len(data)
        self.attributes_values = {}
        self.attributes_values = {}

        for attribute in attributes:
            self.attributes_values[attribute] = {}
            self.attributes_values[attribute]["mean"] = {'YES':0,'NO':0}
            self.attributes_values[attribute]["variance"] = {'YES':0,'NO':0}

        for instance in data:

            # Se calculan frecuencias de los valores del atributo objetivo.
            if instance[target_attr] in self.target_values_frecuency:
                self.target_values_frecuency[instance[target_attr]] += 1
            else:
                self.target_values_frecuency[instance[target_attr]] = 1

            # Se calculan frecuencias de los valores del resto de los atributos.
            for attribute in attributes:

                #Se agregan valores de age para calcular la media y la varianza
                if attribute != target_attr:
                    self.attributes_values[attribute]["mean"][instance[target_attr]] += instance[attribute]
                    self.attributes_values[attribute]["variance"][instance[target_attr]] += instance[attribute]**2

        #Calculo las medias y las varianzas
        for attribute in attributes:
            if attribute != target_attr:
                self.attributes_values[attribute]["mean"]['YES'] = self.attributes_values[attribute]["mean"]['YES'] / self.target_values_frecuency['YES']
                self.attributes_values[attribute]["mean"]['NO'] = self.attributes_values[attribute]["mean"]['NO'] / self.target_values_frecuency['NO']
                self.attributes_values[attribute]["variance"]['YES'] = self.attributes_values[attribute]["variance"]['YES'] / self.target_values_frecuency['YES']
                self.attributes_values[attribute]["variance"]['NO'] = self.attributes_values[attribute]["variance"]['NO'] / self.target_values_frecuency['NO']
                self.attributes_values[attribute]["variance"]['YES'] += -self.attributes_values[attribute]["mean"]['YES']**2
                self.attributes_values[attribute]["variance"]['NO'] += -self.attributes_values[attribute]["mean"]['NO']**2

        # print()
        # print('medias y varianzas NB (calcularlas)')
        # print(self.attributes_values[attribute]["mean"]['YES'])
        # print(self.attributes_values[attribute]["mean"]['NO'])
        # print(self.attributes_values[attribute]["variance"]['YES'])
        # print(self.attributes_values[attribute]["variance"]['NO'])
        # print()

    def classify(self, instance):
        # Clasifica la instancia dada.

        # En result[value] se almacenará  ∏i P(instance[i]|value).P(value)
        # dónde instance[i] es el valor del atributo i en la instancia dada y
        # value es un valor posible del atributo objetivo.
        result = {}

        # Recorre los valores del atributo objetivo.
        # Ej: Pasa por 'YES' y por 'NO'.
        for target_attr_value in self.target_values_frecuency.keys():
            # Se almacena probabilidad del valor objetivo en todo el conjunto.
            # Ej: P(target_attr = 'YES')
            result[target_attr_value] = self.target_values_frecuency[target_attr_value]/self.total
            # Se recorren los atributos de la instancia almacenando su frecuencia.
            # Ej: P(gender = m | target_attr = 'NO')
            for attr in self.attributes:
                if target_attr_value == 'YES':
                    result[target_attr_value] *= self.normal_probability(instance[attr],self.attributes_values[attr]['mean']['YES'],self.attributes_values[attr]['variance']['YES'])
                else:
                    result[target_attr_value] *= self.normal_probability(instance[attr],self.attributes_values[attr]['mean']['NO'],self.attributes_values[attr]['variance']['NO'])

        # Se clasifica según el valor de target_attr con mayor probabilidad.
        return reduce(lambda max_value, value: max_value if result[max_value]>result[value] else value, result.keys())

    def normal_probability(self, value, media , variance):
        if (variance == 0):
            return 1
        else:
            return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

    def holdout_validation(self, validation_set, target_attr):
        # retorna (len(validation_set), cantidad de errores, promedio de errores)
        errors = 0
        for instance in validation_set:
            result = self.classify(instance)
            if result != instance[target_attr]:
                errors += 1

        return (len(validation_set), errors, errors/len(validation_set))
