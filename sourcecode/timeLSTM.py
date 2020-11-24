class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful=False): #Truncated BPTT 역전파를 이용하기 위해 stateful 이용
        self.params = [Wx, Wh, b] # 가중치 매개변수
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] #기울기 초기화
        self.layers = None 

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful # Truncated BPTT를 위해 상태정보 저장

    def forward(self, xs): # 순전파, Xs = [X0, X1, ... , Xt-1]을 의미(한번에 입력)
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f') # 입력과 같은 형상의 출력 hs

        if not self.stateful or self.h is None: # 상태정보가 false이거나 h가 비어있다면,
            self.h = np.zeros((N, H), dtype='f') # 처음 시작한다는 말이므로,
        if not self.stateful or self.c is None: # h와 c를 zero값으로 초기화 
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T): # 시계열 데이터의 크기만큼 반복
            layer = LSTM(*self.params) #LSTM 계층 생성 * T개 만큼
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c) 
            #이전에 저장된 h와 c값(처음이라면 zero), 그리고 입력 xs를 넣어 순전파 
            hs[:, t, :] = self.h # 출력 변수에 저장

            self.layers.append(layer) 

        return hs

    def backward(self, dhs): # 역전파, 이전에서 넘어온 dhs가 인수
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f') # 기울기를 구해서 다음 계층으로 넘기기 위한 그릇
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)): #요소를 뒤집어서 거꾸로 만들고(reversed)
            layer = self.layers[t] # 뒤쪽 계층부터 시작
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc) # 순전파 때 분기했으므로 역전파 때 합산
            dxs[:, t, :] = dx # 다음 계층으로 넘길 기울기
            for i, grad in enumerate(layer.grads): # enumerate는 인덱스값 까지 리턴
                grads[i] += grad #

        for i, grad in enumerate(grads): # enumerate는 인덱스값까지 리턴함
            self.grads[i][...] = grad # 각 계층 기울기 합산해서 최종 기울기를 저장
        self.dh = dh
        return dxs # 하류로 내보내지는 기울기

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None