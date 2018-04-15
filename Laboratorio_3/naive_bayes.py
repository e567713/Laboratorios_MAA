import numpy as np
import math
class NaiveBayes:
	def __init__(self, data, attributes, target_attr):

		self.target_values_frecuency = {}
		self.values_frecuency = {}
		self.target_attr = target_attr
		self.attributes = attributes
		self.total = len(data)

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
			print()
			print('Caclulando prob para: ', target_attr_value)
			print()
			# Probabilidad del valor objetivo en todo el conjunto.
			# Ej: P(target_attr = 'YES')
			result[target_attr_value] = self.target_values_frecuency[target_attr_value]/self.total
			
			# Se recorren los atributos multiplicando la probabilidad anterior por las
			# probabilidades condicionales.
			# Ej: P(gender = m | target_attr = 'NO')
			for attr in self.attributes:
				print()
				print('Atributo: ', attr)
				print(self.values_frecuency[attr])
				result[target_attr_value] *= self.values_frecuency[attr][instance[attr]][target_attr_value] / self.target_values_frecuency[target_attr_value]

		# maximum_prob = 0
		# best_value = 0
		# for target_attr_value in result.keys():
		# 	if result[target_attr_value] > maximum_prob:
		# 		best_value = target_attr_value
		# 		maximum_prob = result[best_value]

		# return best_value
		return 'NO'