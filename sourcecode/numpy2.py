import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.random.randn(10,2)
W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
W2 = np.random.randn(4, 3)
b2 = np.random.randn(3)

h = np.matmul(x, W1) + b1 
a = sigmoid(h) # a는 (10,4) 행렬 
s = np.matmul(a, W2) + b2 
# (10, 4) 행렬과 (4, 3) 행렬의 곱으로 (10, 3) 행렬이 되고
# b2가 (10, 3)으로 브로드 캐스트되어 더해진다.

print(s)
# 즉 출력층은 (10, 3) 행렬의 형상이고
# 2개의 입력을 가진 10쌍의 입력이 3개의 출력을 가진 10쌍의 출력으로 변환된 것을 의미한다.

# 3개의 출력을 가지므로 3개로 분류가 가능하다는 의미
