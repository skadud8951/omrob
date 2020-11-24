import sys, os
sys.path.append(os.pardir) #부모 디렉터리의 파일 가져올 수 있음
from dataset import spiral
import matplotlib.pyplot as plt

x, t = spiral.load_data()
print('x', x.shape) 
print('t', t.shape)