import numpy as np

# 계층 구현 규칙
# 모든 계층은 forward()와 backward() 메서드를 가진다.
# 모든 계층은 인스턴스 변수인 params와 grads를 가진다.

class Sigmoid: #sigmoid 계층
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
        
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out #인스턴스 변수로 저장해서 backward에서 쓰려고
        return out
    
    def backward(self, dout): #dout은 상류에서 하류로 흘러오는 값
        dx = dout * (1.0 - self.out) * self.out # 시그모이드를 미분하면 y(1-y)
        return dx
    
class Affine: # 완전 연결 계층
    def __init__(self, W, b):
        self.params = [W, b] #초기화하면 W, b로 이루어진 리스트가 생성
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None
        
    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out # 입력을 받아서 앞서 배운대로 변환해서 전달
    
    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0) # b가 브로드캐스트 되므로 거꾸로 갈때는 sum
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx
    
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        # 매개 변수를 생성
        
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        # 생성된 매개변수로 신경망을 위한 계층을 생성한 것
        
        self.params = [] # 파라미터 저장을 위한 리스트가 생성되고
        for layer in self.layers:
            self.params += layer.params
            # layer.params는 sigmoid, affine 계층에 있는 params를 의미함
            # Affine.params, Sigmoid.params 겠지..
            # 리스트 값을 더하는게 아니라 결합하는 것임.
            # 따라서 모든 계층의 매개변수가 리스트형태로 결합되어 저장됨
            
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    
x = np.random.randn(10, 2)
model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)