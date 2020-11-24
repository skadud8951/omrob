import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss # 미리 다 추가해놓음

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)
        # 가중치, 편향 초기화
        
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ] # 초기화된 파라미터로 계층 생성
        self.loss_layer = SoftmaxWithLoss() # 손실함수를 계산하는 계층

        self.params, self.grads = [], [] # 계층마다 params와 grads를 모음
        for layer in self.layers:
            self.params += layer.params 
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x) # 각 계층에서 순전파해서 통과
        return x

    def forward(self, x, t): # 순전파하여 통과한 값으로 손실값을 계산
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
