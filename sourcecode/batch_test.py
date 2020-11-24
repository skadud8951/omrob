import numpy as np

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])
y = np.argmax(x, axis = 1) # 열을 기준(axis=1이므로)으로 가장 큰 확률의 인덱스 반환 

print(y) 