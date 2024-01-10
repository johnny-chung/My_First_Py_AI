import numpy as np

#vector
in_vector_2 = np.array([2, 1.5])

weights = np.array([1.45, -0.66])
bias = np.array([0.0])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vt, weights, bias):
    layer_1 = np.dot(input_vt, weights) + bias
    #print(input_vt, " layer 1: ", layer_1)
    layer_2 = sigmoid(layer_1)
    return layer_2




print(in_vector_2, " prediction is ", make_prediction(in_vector_2, weights, bias))


#Gradient_descent

prediction = make_prediction(in_vector_2, weights, bias)
target = 0

mse = np.square(prediction - target)
print("\n================")
print(f"prediction: {prediction} | Error: {mse}")


derivative = 2 * (prediction - target)
print(f"derivative: {derivative}" )

new_weights = weights - derivative
print("new weight:", new_weights)
def make_pred_and_print(input_vt, weights, bias):
    prediction = make_prediction(input_vt, weights, bias)
    error = np.square(prediction - target)
    print(f"new prediction: {prediction} | Error: {error}\n")

make_pred_and_print(in_vector_2, new_weights, bias)

# add alpha
alpha = 0.1
new_weights = weights - alpha * derivative
print("\n ------ with alpha -------")
print(f"alpha: {alpha}")
make_pred_and_print(in_vector_2, new_weights, bias)

alpha = 0.01
new_weights = weights - alpha * derivative
print(f"alpha: {alpha}")
make_pred_and_print(in_vector_2, new_weights, bias)
alpha = 0.001
new_weights = weights - alpha * derivative
print(f"alpha: {alpha}")
make_pred_and_print(in_vector_2, new_weights, bias)

