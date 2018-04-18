import numpy as np
import math
from functools import reduce

class NaiveBayes:
	def __init__(self, data, attributes, target_attr):

		self.target_values_frecuency = {}
		self.values_frecuency = {}
		self.target_attr = target_attr
		self.attributes = attributes
		self.total = len(data)
		self.m = 0.5

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
						self.values_frecuency[attribute][instance[attribute]]['NO'] = 1


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

				# Si encuentro un valor nunca visto anteriormente en el atributo attr,
				# se lo agrega con frecuencia 0 (es agregado para tenerlo en cuenta a la 
				# hora de calcular la prioridad p del estimador m en esta y futuras clasificaciones).
				if not instance[attr] in self.values_frecuency[attr]:
					self.values_frecuency[attr][instance[attr]] = {'YES':0, 'NO':0}

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