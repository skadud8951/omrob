import sys
sys.path.append('..')
from common.time_layers import *
from common.np import *  # import numpy as np
from common.base_model import BaseModel
from common.util import *
import pickle


class Rnnlm(BaseModel):
    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 가중치를 랜덤으로 초기화
        embed_W = (rn(V, D) / 100).astype('f') # 단어를 벡터로 변환하는 embedded층, 단어 입력만큼 V*D 형상
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f') # 가중치 Wx와 Wh 
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f') # 편향 
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f') # affine 계층 생성
        affine_b = np.zeros(V).astype('f')

        # 계층 생성
        self.layers = [ #순서대로 생성
            TimeEmbedding(embed_W), 
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True), # Truncated BPTT를 위해 
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss() # softmaxwithLoss층은 따로 생성
        self.lstm_layer = self.layers[1] # reset_state를 위해

        # 모든 가중치와 기울기를 리스트에 모음
        self.params, self.grads = [], [] # 가중치와 기울기를 담을 리스트 생성
        for layer in self.layers:
            self.params += layer.params # 가중치 다 더해서 인스턴트 변수에 저장
            self.grads += layer.grads # 기울기 다 더해서 인스턴트 변수에 저장

    def predict(self, xs): 
        for layer in self.layers:
            xs = layer.forward(xs) #timesoftmaxwithLoss 층을 제외하고 순전파 
        return xs

    def forward(self, xs, ts): 
        score = self.predict(xs) #loss_layer 이전까지 다 구해서 score에 저장
        loss = self.loss_layer.forward(score, ts) 
        return loss

    def backward(self, dout=1): 
        dout = self.loss_layer.backward(dout) # loss_layer 우선 역전파 실시
        for layer in reversed(self.layers): #reversed해서 거꾸로 진행
            dout = layer.backward(dout) # 거꾸로 역전파 진행
        return dout

    def reset_state(self): # 신경망 상태를 초기화
        self.lstm_layer.reset_state()

class BetterRnnlm(BaseModel):
    '''
     LSTM 계층을 2개 사용하고 각 층에 드롭아웃을 적용한 모델이다.
     아래 [1]에서 제안한 모델을 기초로 하였고, [2]와 [3]의 가중치 공유(weight tying)를 적용했다.

     [1] Recurrent Neural Network Regularization (https://arxiv.org/abs/1409.2329)
     [2] Using the Output Embedding to Improve Language Models (https://arxiv.org/abs/1608.05859)
     [3] Tying Word Vectors and Word Classifiers (https://arxiv.org/pdf/1611.01462.pdf)
    '''
    def __init__(self, vocab_size=10000, wordvec_size=650,
                 hidden_size=650, dropout_ratio=0.5):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx1 = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh1 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b1 = np.zeros(4 * H).astype('f')
        lstm_Wx2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_Wh2 = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b2 = np.zeros(4 * H).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding(embed_W),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx1, lstm_Wh1, lstm_b1, stateful=True),
            TimeDropout(dropout_ratio),
            TimeLSTM(lstm_Wx2, lstm_Wh2, lstm_b2, stateful=True),
            TimeDropout(dropout_ratio),
            TimeAffine(embed_W.T, affine_b)  # weight tying!!
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layers = [self.layers[2], self.layers[4]]
        self.drop_layers = [self.layers[1], self.layers[3], self.layers[5]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs, train_flg=False):
        for layer in self.drop_layers:
            layer.train_flg = train_flg

        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts, train_flg=True):
        score = self.predict(xs, train_flg)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        for layer in self.lstm_layers:
            layer.reset_state()
