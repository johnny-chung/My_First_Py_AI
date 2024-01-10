import numpy as np

#vector
in_vector_1 = np.array([1.66, 1.56])
in_vector_2 = np.array([2, 1.5])

weights = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vt, weights, bias):
    layer_1 = np.dot(input_vt, weights) + bias
    print(input_vt, " layer 1: ", layer_1)
    layer_2 = sigmoid(layer_1)
    return layer_2

print(in_vector_1, " prediction is ", make_prediction(in_vector_1, weights, bias))
print(in_vector_2, " prediction is ", make_prediction(in_vector_2, weights, bias))