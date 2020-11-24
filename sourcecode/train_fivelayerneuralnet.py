import sys, os
sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
from five_layer_net import FiveLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = FiveLayerNet(input_size=784, hidden1_size=50, hidden2_size=50, hidden3_size=50, hidden4_size=50, output_size=10)
# 레이어 수에 맞게 hidden4 까지 값을 설정해줌 

iters_num = 10000 # 갱신 반복 횟수
train_size = x_train.shape[0] 
batch_size = 100 # 미니 배치 크기
learning_rate = 0.01 # 학습률

train_loss_list = [] # 손실 함수 값의 변화 추이 비교 목적
train_acc_list = [] # 정확도 평가를 위함
test_acc_list = [] # training 정확도와 비교 목적
activations = {}

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size) # 설정한 사이즈만큼 랜덤한 dataset 설정
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # 기울기 계산
    #grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch) # 오차역전파법 방식
    
    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3', 'W4', 'b4', 'W5', 'b5'):
        network.params[key] -= learning_rate * grad[key]
    
    loss = network.loss(x_batch, t_batch) # 손실 함수 값을 구함
    train_loss_list.append(loss) #손실 함수 값을 리스트로 저장
    
    if i % iter_per_epoch == 0:# 에폭당 정확도를 계산해서 값이 어떻게 변화하는지 확인
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)