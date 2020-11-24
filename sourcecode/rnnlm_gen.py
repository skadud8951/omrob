import sys
sys.path.append('..')
import numpy as np
from common.functions import softmax
from rnnlm import Rnnlm
from rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm): # Rnnlm을 상속하는 class 
    def generate(self, start_id, skip_ids=None, sample_size=100): 
        #start_id: 최초로 주는 단어의 ID, skip_ids: 샘플링 하고싶지않은 단어의 ID
        #Sample_size: 샘플링하는 단어의 수
        word_ids = [start_id] #샘플링해서 점점 id를 추가할 예정

        x = start_id 
        while len(word_ids) < sample_size: #리스트 갯수를 100개 이하로 한정
            x = np.array(x).reshape(1, 1) #2차원배열로 reshape
            score = self.predict(x) #각 단어별 점수를 만듬(정규화 되기 전)
            p = softmax(score.flatten())  # 소프트맥스 함수로 정규화 (확률분포가 됨)

            sampled = np.random.choice(len(p), size=1, p=p) 
            #확률분포를 통해 높은확률의 단어는 자주 선택되고 낮은확률의 단어는 희소하게 선택됨(랜덤으로)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x)) # 차례차례 선택된 id 추가

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)
        
