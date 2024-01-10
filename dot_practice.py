import numpy as np

in_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

dot_p_1 = np.dot(in_vector, weights_1)
dot_p_2 = np.dot(in_vector, weights_2)

print(dot_p_1, dot_p_2)