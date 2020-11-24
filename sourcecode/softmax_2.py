import numpy as np
import matplotlib as plt

# 지수가 너무 커서 overflow발생 
a = np.array([1010, 1000, 990])
y= np.exp(a) / np.sum(np.exp(a))
print(y) 

# 입력 신호 중 최댓값을 이용하여 보정
c = np.max(a)
a = a - c

y = np.exp(a) / np.sum(np.exp(a))
print(y) # 보정해도 결과는 변하지 않음

# 함수로 정리
def softmax(a): 
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    
    return exp_a / sum_exp_a

print(softmax(a))

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y)) #softmax 함수의 출력 총합은 1.0

