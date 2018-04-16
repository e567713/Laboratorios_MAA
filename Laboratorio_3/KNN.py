import math
import numpy as np

def get_k_nearest(instance, k, data):
  distances_array = []
  k_nearest = []
  i = -1
  for example in data:
    i += 1
    distances_array.append((i, distance(instance, example)))
  distances_array.sort(key = lambda tup: tup[1])
  print (distances_array)
  # HAY QUE PROBAR SI ESTO ANDA




def distance(instance1, instance2):
  instance1_values = list(instance1.values())
  instance2_values = list(instance2.values())
  sumatory = 0
  for i in range (0, len(instance1_values)):
      if isinstance(instance1_values[i], np.float64):
        sumatory += pow(abs(instance2_values[i] - instance1_values[i]), 2)
      elif instance1_values[i] != instance2_values[i]:
        sumatory += 1

  return math.sqrt(sumatory)