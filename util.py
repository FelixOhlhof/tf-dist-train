import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Sequential
from keras.layers import Dense
from keras.models import clone_model
from numpy import average
from numpy import array
import sys, pickle #just for debugging

m = ""

# create a model from the weights of multiple models
def model_weight_ensemble(members):
	# prepare an array of equal weights
	n_models = len(members)
	weights = [1/n_models for i in range(1, n_models+1)]
	

	# determine how many layers need to be averaged
	n_layers = len(members[0].get_weights())
	# create an set of average model weights
	avg_model_weights = list()
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = array([model.get_weights()[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		avg_model_weights.append(avg_layer_weights)
	# create a new model with the same structure
	model = clone_model(members[0])
	# set the weights in the new
	model.set_weights(avg_model_weights)
	model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
	return model

def get_average_weights(members):
	# prepare an array of equal weights
	n_models = len(members)
	weights = [1/n_models for i in range(1, n_models+1)]
	new_weights = members[0]

	# determine how many layers need to be averaged
	n_layers = len(members[0])
	
	for layer in range(n_layers):
		# collect this layer from each model
		layer_weights = array([model[layer] for model in members])
		# weighted average of weights for this layer
		avg_layer_weights = average(layer_weights, axis=0, weights=weights)
		# store average layer weights
		new_weights[layer] = avg_layer_weights

		# f = open("sample_org.txt", "wb")
		# f.write(pickle.dumps(new_weights))
		# f.close()
	return new_weights
 