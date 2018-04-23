# coding=utf-8
import numpy as np
import math
from functools import reduce

class NaiveBayes:
	def __init__(self, data, attributes, target_attr):

		self.target_values_frecuency = {}
		self.values_frecuency = {}
		self.normalize_age_probabilities = {}
		self.target_attr = target_attr
		self.attributes = attributes
		self.total = len(data)
		self.m = 0.5
		self.count_yes=0
		self.count_no=0

		self.normalize_age_probabilities["media"] = {'YES':0,'NO':0}
		self.normalize_age_probabilities["variance"] = {'YES':0,'NO':0}

		for attribute in attributes:
			self.values_frecuency[attribute] = {}
		for instance in data:

			# Se calculan frecuencias de los valores del atributo objetivo.
			if instance[target_attr].decode() in self.target_values_frecuency:
				self.target_values_frecuency[instance[target_attr].decode()] += 1
			else:
				self.target_values_frecuency[instance[target_attr].decode()] = 1
			
			# Se calculan frecuencias de los valores del resto de los atributos.
			for attribute in attributes:

				# Si ya se han contado ejemplos para ese valor se suma 1 a su frecuencia
				# en el valor del atributo objetivo correspondiente.
				if instance[attribute] in self.values_frecuency[attribute]:
					if instance[target_attr].decode() == 'YES':
						self.values_frecuency[attribute][instance[attribute]]['YES'] += 1
					elif instance[target_attr].decode() == 'NO':
						self.values_frecuency[attribute][instance[attribute]]['NO'] += 1
				
				# En caso contrario, primero se crea la entrada de ese valor, para luego
				# colocarle el contador en 1 en dónde corresponda.
				else:
					self.values_frecuency[attribute][instance[attribute]]= {'YES':0,'NO':0}
					if instance[target_attr].decode() == 'YES':
						self.values_frecuency[attribute][instance[attribute]]['YES'] = 1
					elif instance[target_attr].decode() == 'NO':
						# print (attribute)
						# print (instance[attribute])
						self.values_frecuency[attribute][instance[attribute]]['NO'] = 1
						# print ("PASOOO")

			#Se agregan valores de age para calcular la media y la varianza
			if instance[target_attr].decode() == 'YES':
				self.count_yes += 1
				self.normalize_age_probabilities["media"]['YES'] += instance["age"]
				self.normalize_age_probabilities["variance"]['YES'] += instance["age"]**2
			if instance[target_attr].decode() == 'NO':
				self.count_no += 1
				self.normalize_age_probabilities["media"]['NO'] += instance["age"]
				self.normalize_age_probabilities["variance"]['NO'] += instance["age"]**2

		self.normalize_age_probabilities["media"]['YES'] = self.normalize_age_probabilities["media"]['YES'] / self.count_yes
		self.normalize_age_probabilities["media"]['NO'] = self.normalize_age_probabilities["media"]['NO'] / self.count_no
		self.normalize_age_probabilities["variance"]['YES'] = self.normalize_age_probabilities["variance"]['YES'] / self.count_yes
		self.normalize_age_probabilities["variance"]['NO'] = self.normalize_age_probabilities["variance"]['NO'] / self.count_no
		self.normalize_age_probabilities["variance"]['YES'] += -self.normalize_age_probabilities["media"]['YES']**2
		self.normalize_age_probabilities["variance"]['NO'] += -self.normalize_age_probabilities["media"]['NO']**2

	def classify(self, instance, normalize):
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

				if attr=="age" and normalize:
					if target_attr_value == 'YES':
						result[target_attr_value] *= self.normal_probability(instance[attr],self.normalize_age_probabilities['media']['YES'],self.normalize_age_probabilities['variance']['YES'])
					else:
						result[target_attr_value] *= self.normal_probability(instance[attr],self.normalize_age_probabilities['media']['NO'],self.normalize_age_probabilities['variance']['NO'])

				elif instance[attr] in self.values_frecuency[attr]:

					# Se multiplica la aproximación al valor que se viene calculando.
					result[target_attr_value] *= self.m_estimate(instance, attr, target_attr_value)

		# Se clasifica según el valor de target_attr con mayor probabilidad.
		return reduce(lambda max_value, value: max_value if result[max_value]>result[value] else value, result.keys())

	def m_estimate(self, instance, attr, target_attr_value):
		# OBS: Asume prioridad apriori con distribución uniforme.

		# Ej: P(attr = instance[attr] | target_attr = target_attr_value)
		numerator = self.values_frecuency[attr][instance[attr]][target_attr_value]
		
		# Se le suma m*p (dónde p = 1/|valores posibles de attr|) 
		numerator += self.m * (1/len(self.values_frecuency[attr]))

		# Ej: P(target_attr = target_attr_value)
		denominator = self.target_values_frecuency[target_attr_value]

		# Se le suma m
		denominator += self.m

		return numerator / denominator

	def normal_probability(self, value, media , variance):
		return (1 / math.sqrt(2*math.pi*variance))*math.e**((-((value-media)**2))/(2*variance))

