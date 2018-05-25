import random
from math import e 
class Neuron:
	def __init__(self):
		self.weights = []

	def create(self, n_inputs):
		# Inicializa los pesos de la neurona con valores entre -0.5 y 0.5 de forma aleatoria.
		self.weights = [round(random.uniform(-0.5,0.5),5) for i in range(n_inputs)]
	
	def compute_input(self, inputs):
		# Calcula la combinación lineal entre los pesos y los valores de las inputs.
		# Asume que el valor w0 es el último en la lista de pesos.
		result = self.weights[-1]
		for i in range(len(self.weights)-1):
			result += self.weights[i] * inputs[i]
		return result

	def apply_sigmoid(self, computed_value):
		# Aplica la función de sigmoide.
		return 1 / (1 + e**(-computed_value))