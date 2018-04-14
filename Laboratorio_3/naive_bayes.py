class NaiveBayes:
    def __init__(self, data, attributes, target_attr):
    	target_values_probabilities = []
    	
    	self.target_values_frecuency = {}
    	self.values_frecuency = {}	
    	
    	for instance in data:
    		if self.target_values_frecuency tiene esta key -> instance[target_attr]:
    			self.target_values_frecuency[instance[target_attr]] += 1
    		else:
    			self.target_values_frecuency agregar key instance[target_attr] con valor 1

    	for attribute in attributes:
    		self.values_frecuency agrego key attribute con valor {}

    	for instance in data:
	    	for attribute in attributes:	
	    		if instance[attribute] in self.values_frecuency[attribute]:
	    			sumamos 1 a la coordenada dependiendo de si es YES o NO
	    		else:
	    			instance[target_attr] = 'YES' (1,0)


    		
    	self.values_frecuency: {
    		'Viento': {
    			'Fuerte': ( , 0.56) #Primer coordenada corresponde YES
    			'Debil': ( , )
    		}
    	}

    def classify(self, instance):
    	self.target_values_frecuenc