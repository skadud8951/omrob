import numpy as np # numpy를 np라는 이름으로 사용
from dataset.mnist import load_mnist # mnist dataset 불러오기
from two_layer_net import TwoLayerNet # 만들어뒀던 2층 신경망 사용

# mnist dataset 읽기, 데이터를 0.0 ~ 1.0 범위의 데이터로 정규화를 하고, 
# 정답 원소만 1값을 가지는 원-핫 인코딩 사용
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# 이전에 구현한 2층 신경망 생성
network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 갱신을 반복하는 횟수
train_size = x_train.shape[0]
batch_size = 100   # 미니배치 크기
learning_rate = 0.1 # 학습률 설정

train_loss_list = []  # 손실 함수 값의 추이를 확인하기 위해 손실 값을 담는 리스트를 생성
train_acc_list = [] # 에폭을 반복할 때마다 training dataset의 accuracy를 계산하여 담는 리스트를 생성
test_acc_list = [] # 에폭을 반복할 때마다 test dataset의 accuracy를 계산하여 담는 리스트를 생성

# 1에폭당 학습수(6만개의 training set 중 100개의 미니배치 => 1에폭은 미니배치의 600번 반복)
iter_per_epoch = max(train_size / batch_size, 1)

# 매개변수 갱신 과정
for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    # 6만개의 training dataset 중 100개의 dataset을 무작위로 선정(임의의 정수)
    x_batch = x_train[batch_mask] # 훈련 데이터 미니 배치 획득
    t_batch = t_train[batch_mask] # 정답 레이블 미니 배치 획득
    
    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)
    # 훈련 데이터와 정답 레이블을 비교하여 손실 함수를 계산하고, 
    # 손실 함수에 대한 매개 변수 기울기를 각각 구함. 
    
    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
    # 학습률(갱신하는 정도)을 기울기에 곱하여 원래 매개변수 값에서 뺌
    # 변화 방향은 손실 함수를 낮추는 방향이므로 갱신할수록 손실 함수가 작아진다.
    
    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch) # 손실 함수 값을 구한다.
    train_loss_list.append(loss)
    # 각 반복마다 손실 함수 값을 리스트로 이어 붙여 추후에 도식적으로 확인하기 위함
    
    # 1에폭당 정확도 계산(600번째 갱신마다 정확도 계산)
    if i % iter_per_epoch == 0: 
        train_acc = network.accuracy(x_train, t_train) # training set의 정확도를 구함
        test_acc = network.accuracy(x_test, t_test) # test set의 정확도를 구함
        train_acc_list.append(train_acc) # 도식화를 위해 리스트에 정확도를 넣음
        test_acc_list.append(test_acc) # 도식화를 위해 리스트에 정확도를 넣음
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

