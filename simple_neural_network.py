#{?}
# Build a neural network in 4 minutes
#     -> Siraj Raval

import tensorflow as tf
import numpy as np

def nonlin(x, deriv=False): #sigmoid function
    if (deriv == True):
        return x * (1 - x)

    return (1 / (1 + np, exp(-x)))

#input data
input_x = tf.Variable.([[0.0,1],
                  [0,1,1],
                  [1,0,1],
                  [1,1,1]])
#input_x = tf.tensordot(tf.zeros([4,3], tf.int32), input_x, axes=[[1], [0]])
print(input_x)
#output data
labels_y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

#synapses
syn0 = tf.random.random((4,3))# 3x4 matrix of random weights
syn1 = tf.random.random((3,1)) # 4x1 matrix of random weights

#training step
num_epochs = 60000
for i in range(0, num_epochs):
    input_layer = input_x #represents dementionality of inputs
    layer_one = nonlin(np.dot(input_layer, syn0))
        #applies activation over first layer (dot product)
    output_layer = nonlin(np.dot(layer_one, syn1))
        #the last nonlin(weights, biases), represents shape of output

    output_err = labels_y - output_layer
        #determine error in prediction after epoch

    if (i % 10000) == 0:
        print('Error:' + str(np.mean(np.abs(output_err))))

    #Backpropegation:
    output_layer_delta = ouput_errr * nonlin(layer_one, derv=True)
        #compute slope of least error model
    layer_one_error = output_layer_delta.dot(syn1.T)
        #get error in first layer by dot product transposed
    
    layer_one_delta = layer_one_error * nonlin(layer_one,deriv=True)
        #compute slope of least error model, relative to previous layer

    #update weights
    syn1 = layer_one.T.dot(layer_one_delta)
    syn0 = input_layer.T.dot(output_layer)

print("Output after training:", output_layer)

    

    
