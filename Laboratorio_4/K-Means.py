import random
import math
class KMeans:
	def __init__(self, k, max_iter, times):
		# k: Número de clusters. 
		# max_iter: Máximo de iteraciones (si no se converge).
		# times: Cantidad de veces que se corre el algoritmo con distintas inicializaciones.
		self.k = k
		self.max_iter = max_iter
		self.times = times

		self.centroids = []

		self.optimal = None

	
	def train(self, data):

		for i in range(self.times):

			# Selecciono k centroides iniciales al azar para la corrida i del algoritmo.
			centroids = random.sample(data, self.k)

			# Los asigno aleatoriamente al conjunto de entrenamiento
			assigned_centroids = [random.choice(range(k)) for iter in range(k)]

			j = 0
			while not self.has_converged() and j < max_iter: 	
				
				# Recorro conjunto de entrenamiento.
				for index, instance in data:

					# Asigno centroides a los ejemplos.
					assigned_centroids[index] = self.assign(instance, centroids)
					
					# Actualizo lista de centroides. 
					self.update()



			self.centroids.append(current_centroids)



	def assign(self, instance, centroids):
		min_distance = self.euclidean_distance(instance, centroids[0])
		min_centroid = centroids[0]

		for i in range (1, len(centroids)):
			new_distance = self.euclidean_distance(instance, centroids[i])
			if new_distance < min_distance:
				min_distance = new_distance
				min_centroid = centroids[i]

		return min_centroid

	def update

	def euclidean_distance(X, Y):
		distance = 0
		for i in range(X):
			distance += (X[i] + Y[i])**2
		return math.sqrt(distance)

