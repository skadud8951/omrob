import numpy as np
import matplotlib as plt

a = np.array([0.3, 2.9, 4.0])

exp_a = np.exp(a)
sum_exp_a = np.sum(a)
y = exp_a / sum_exp_a
print(y)

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(a)
    return exp_a / sum_exp_a
    
print(softmax(a))