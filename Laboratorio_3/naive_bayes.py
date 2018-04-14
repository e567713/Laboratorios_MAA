import numpy as np
import math
class NaiveBayes:
	def __init__(self, data, attributes, target_attr):
		# target_values_probabilities = []
		self.target_values_frecuency = {}
		self.values_frecuency = {}
		for attribute in attributes:
			self.values_frecuency[attribute] = {}
		for instance in data:
			if instance[target_attr] in self.target_values_frecuency:
				self.target_values_frecuency[instance[target_attr]] += 1
			else:
				self.target_values_frecuency[instance[target_attr]] = 1
			#SE PIERDE LA GENERALIDAD CON EL TARGET ATTRIBUTE
			for attribute in attributes:
				# print(attribute)
				# print(instance[attribute])
				# print(self.values_frecuency[attribute])
				if instance[attribute] in self.values_frecuency[attribute]:
					if instance[target_attr]==b'YES':
						self.values_frecuency[attribute][instance[attribute]][b'YES'] += 1
					elif instance[target_attr]==b'NO':
						self.values_frecuency[attribute][instance[attribute]][b'NO'] += 1
				else:
					# print(self.values_frecuency[attribute][instance[attribute]])
					self.values_frecuency[attribute][instance[attribute]]= {b'YES':0,b'NO':0}
					if instance[target_attr]==b'YES':
						self.values_frecuency[attribute][instance[attribute]][b'YES'] = 1
					elif instance[target_attr]==b'NO':
						self.values_frecuency[attribute][instance[attribute]][b'NO'] = 1

#   	for instance in data:
#    	for attribute in attributes:
#    		if instance[attribute] in self.values_frecuency[attribute]:
#    			sumamos 1 a la coordenada dependiendo de si es YES o NO
#    		else:
#    			instance[target_attr] = 'YES' (1,0)
#
#
#
# 	self.values_frecuency: {
# 		'Viento': {
# 			'Fuerte': ( , 0.56) #Primer coordenada corresponde YES
# 			'Debil': ( , )
# 		}
# 	}
#
# def classify(self, instance):
# 	self.target_values_frecuenc
