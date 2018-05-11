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
            self.attributes_values[attribute]["mean"] = {b'YES':0,b'NO':0}
            self.attributes_values[attribute]["variance"] = {b'YES':0,b'NO':0}

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
                self.attributes_values[attribute]["mean"][b'YES'] = self.attributes_values[attribute]["mean"][b'YES'] / self.target_values_frecuency[b'YES']
                self.attributes_values[attribute]["mean"][b'NO'] = self.attributes_values[attribute]["mean"][b'NO'] / self.target_values_frecuency[b'NO']
                self.attributes_values[attribute]["variance"][b'YES'] = self.attributes_values[attribute]["variance"][b'YES'] / self.target_values_frecuency[b'YES']
                self.attributes_values[attribute]["variance"][b'NO'] = self.attributes_values[attribute]["variance"][b'NO'] / self.target_values_frecuency[b'NO']
                self.attributes_values[attribute]["variance"][b'YES'] += -self.attributes_values[attribute]["mean"][b'YES']**2
                self.attributes_values[attribute]["variance"][b'NO'] += -self.attributes_values[attribute]["mean"][b'NO']**2

    def classify(self, instance):
        # Clasifica la instancia dada.

        # En result[value] se almacenará  ∏i P(instance[i]|value).P(value)
        # dónde instance[i] es el valor del atributo i en la instancia dada y
        # value es un valor posible del atributo objetivo.
        result = {}

        # Recorre los valores del atributo objetivo.
        # Ej: Pasa por b'YES' y por b'NO'.
        for target_attr_value in self.target_values_frecuency.keys():
            # Se almacena probabilidad del valor objetivo en todo el conjunto.
            # Ej: P(target_attr = b'YES')
            result[target_attr_value] = self.target_values_frecuency[target_attr_value]/self.total
            # Se recorren los atributos de la instancia almacenando su frecuencia.
            # Ej: P(gender = m | target_attr = b'NO')
            for attr in self.attributes:
                if target_attr_value == b'YES':
                    result[target_attr_value] *= self.normal_probability(instance[attr],self.attributes_values[attr]['mean'][b'YES'],self.attributes_values[attr]['variance'][b'YES'])
                else:
                    result[target_attr_value] *= self.normal_probability(instance[attr],self.attributes_values[attr]['mean'][b'NO'],self.attributes_values[attr]['variance'][b'NO'])

        # Se clasifica según el valor de target_attr con mayor probabilidad.
        return reduce(lambda max_value, value: max_value if result[max_value]>result[value] else value, result.keys())

    def normal_probability(self, value, media , variance):
        return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))
