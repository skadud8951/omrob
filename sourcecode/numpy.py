import numpy as np

x = np.array([1, 2, 3]) # 행벡터
y = np.array([[1],[2],[3]]) # 열벡터
print(x)
print(y)
print(x.__class__) # 클래스 이름 표시
print(x.shape) # (3,)로 출력
print(y.shape) # (1,3)으로 출력
print(x.ndim) # 1로 출력 => 1차원(한 줄로 늘어서 있으므로)
print(y.ndim) # 2차원으로 본다

W = np.array([[1, 2, 3],[4, 5, 6]])
print(W)
print(W.shape) # (2,3)
print(W.ndim) # 2차원

W = np.array([[1,2,3], [4,5,6]])
X = np.array([[0,1,2], [3,4,5]])
print(W+X) # 원소별 연산
print(W*X) # 원소별 연산

A = np.array([[1,2], [3,4]])
print(A * 10) # 브로드캐스트되어 원소별 연산함

a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a,b)) # 내적

A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(np.matmul(A, B)) # 행렬의 곱 

################

W1 = np.random.randn(2, 4)
b1 = np.random.randn(4)
x = np.random.randn(10, 2) #10개의 샘플 데이터를 이용 
h = np.matmul(x, W1) + b1 
# (10,2) 행렬과 (2, 4) 행렬이 곱해져 (10,4) 행렬이 만들어지고 
# b1은 (10, 4)로 브로드캐스트 되어 더해진다.

