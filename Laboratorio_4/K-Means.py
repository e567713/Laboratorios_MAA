import random
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
			initial_centroids = random.sample(data, self.k)

			# Los asigno aleatoriamente al conjunto de entrenamiento
			current_centrois = []

			j = 0
			while not self.has_converged() and j < max_iter: 	
				
				# Recorro conjunto de entrenamiento.
				for instance in data:

					current_centroids[] = self.assign()
					self.update()



			self.centroids.append(current_centroids)



	def assign

	def update

