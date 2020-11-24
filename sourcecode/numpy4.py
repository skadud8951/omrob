import numpy as np

D, N = 8, 7
x = np.random.randn(1, D)
y = np.repeat(x, N, axis=0) # x를 N개로 분기
dy = np.random.randn(N, D) # 기울기를 그냥 무작위로 설정한것
dx = np.sum(dy, axis=0, keepdims=True) # 역전파에서는 총 합이므로

print(dx)

D, N = 8, 7
x = np.random.randn(N, D)
y = np.sum(x, axis=0, keepdims=True)

dy = np.random.randn(1, D) # 기울기 무작위로 설정
dx = np.repeat(dy, N, axis=0) 