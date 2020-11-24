class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr # lr은 학습률
        
    def update(self, params, grads):
        for i in range(len(params)): # 첫번째 차원의 행 수 반환
            params[i] -= self.lr * grads[i] 
            # 미니배치된 입력 데이터마다 파라미터와 기울기가 존재하므로
            
