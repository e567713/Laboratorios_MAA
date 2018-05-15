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

		self.optimal = None

	
	def train(self, data):

		intermediate_results = []

		for i in range(self.times):

			# Selecciono k centroides iniciales al azar para la corrida i del algoritmo.
			centroids = random.sample(data, self.k)

			# Asigno los centroides aleatoriamente al conjunto de entrenamiento
			assigned_centroids = [random.choice(range(k)) for iter in range(k)]

			j = 1
			past_assigned_centroids = []
			while past_assigned_centroids != assigned_centroids and j <= max_iter: 	
				past_assigned_centroids = assigned_centroids
				j += 1

				# Recorro conjunto de entrenamiento.
				for index, instance in data:

					# Asigno centroides a los ejemplos.
					assigned_centroids[index] = self.assign(instance, centroids)
				
				# Recorro los centroides.
				for index_to_update, centroid_to_update in centroids:

					# Actualizo lista de centroides. 
					centroids[index_to_update] = self.update(index_to_update, centroid_to_update, assigned_centroids, data)

			# Almaceno los centroides hallados junto con sus asignaciones en cada corrida del algoritmo.
			intermediate_results.append((centroids, assigned_centroids))

		# Nos quedamos con el resultado óptimo medido a través de la distancia euclidiana.
		self.optimal = self.get_best_result(intermediate_results, data)

	def assign(self, instance, centroids):
		# Devuelve el índice del centroide cuya distancia es menor al ejemplo dado.

		min_distance = self.euclidean_distance(instance, centroids[0])
		min_index = 0

		for i in range (1, len(centroids)):
			new_distance = self.euclidean_distance(instance, centroids[i])
			if new_distance < min_distance:
				min_distance = new_distance
				min_index = i

		return min_index

	def update(self, index_to_update, centroid_to_update, assigned_centroids, data):
		# Actualiza el centroide centroid_to_update ubicado en el índice index_to_update dentro
		# de la lista de centroides.

		numerator = centroid_to_update

		# Recorro lista de centroides asignados.
		for index, assigned_centroid in assigned_centroids:

			# Si un ejemplo tiene asignado el centroide a actualizar.
			if index_to_update == assigned_centroid:

				# Sumo el ejemplo que posee el centroide a actualizar.
				numerator = self.vector_sum(data[index], numerator)
				denominator += 1

		return map(lambda elem: elem/denominator, numerator)

	def euclidean_distance(X, Y):
		return math.sqrt([(x - y)**2 for x, y in zip(X, Y)])

	def vector_sum(X, Y):
		# Suma los vectores X e Y.
		return [x + y for x, y in zip(X, Y)]

	def get_best_result(self, intermediate_results, data):
		best_result_index = 0
		centroids = intermediate_results[0][0] 
		assigned_centroids = intermediate_results[0][1] 
		
		best_result = 0
		for index, instance in data:
			best_result += self.euclidean_distance(instance, centroids[assigned_centroids[index]])


		for i in range(1, len(intermediate_results)):
			centroids = intermediate_results[i][0] 
			assigned_centroids = intermediate_results[i][1] 

			result = 0
			for index, instance in data:
				result += self.euclidean_distance(instance, centroids[assigned_centroids[index]])

			if result < best_result:
				best_result = result
				best_result_index = i

		return intermediate_results[best_result_index]