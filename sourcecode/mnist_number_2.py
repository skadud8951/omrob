import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from PIL import Image

def sigmoid(x):
    return 1/(1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    return exp_a / sum_exp_a

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img)) #데이터를 넘파이배열로 만든 뒤 fromarray를 통해 이미지로 변환
    pil_img.show()

def get_data(): # 입력 데이터 설정
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network(): # 가중치, 편향 
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f) #이미 학습이 되어있는 3층 신경망 sample 가중치 파일을 사용
    
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1) +b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) +b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) +b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in range(0,len(x),batch_size): #0부터 len(x)-1까지 batch_size의 간격으로 증가
    x_batch = x[i:i+batch_size] #리스트 슬라이싱 (x[0]부터 x[99]까지) 
    y_batch = predict(network, x_batch)
    p  = np.argmax(y_batch, axis = 1) # 열 기준(axis=1) 가장 높은 확률 인덱스 반환
    accuracy_cnt += np.sum(p == t[i:i+batch_size]) # true갯수 합만큼 cnt
        
print("Accuracy:"+ str(float(accuracy_cnt)/len(x)))