import numpy as np
import math
class NaiveBayes:
	def __init__(self, data, attributes, target_attr):

		self.target_values_frecuency = {}
		self.values_frecuency = {}

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

				# Si ya se han contado ejemplos para ese valor se suma uno a su frecuencia
				# en el valor del atributo objetivo correspondiente.
				if instance[attribute] in self.values_frecuency[attribute]:
					if instance[target_attr]==b'YES':
						self.values_frecuency[attribute][instance[attribute]]['YES'] += 1
					elif instance[target_attr]==b'NO':
						self.values_frecuency[attribute][instance[attribute]]['NO'] += 1
				
				# En caso contrario, primero se crea la entrada de ese valor, para luego
				# colocarle el contador en 1 en dónde corresponda.
				else:
					self.values_frecuency[attribute][instance[attribute]]= {'YES':0,'NO':0}
					if instance[target_attr]==b'YES':
						self.values_frecuency[attribute][instance[attribute]]['YES'] = 1
					elif instance[target_attr]==b'NO':
						self.values_frecuency[attribute][instance[attribute]]['NO'] = 1


	def classify(self, instance):
		# TODO
		return 'YES'
