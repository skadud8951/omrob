#1. Import the numpy package under the name np
import numpy as np

#2. Print the numpy version and the configuration
print(np.__version__)
np.show_config()

#3. Create a null vector of size 10
A = np.zeros(10)
print(A)

#4. How to find the memory size of any array
print(A.size*A.itemsize) # 10*8byte

#5. How to get the documentation of the numpy add function from the command line?
%run 'python -c "import numpy; numpy.info(numpy.add)"'

#6. Create a null vector of size 10 but the fifth value which is 1
A = np.zeros(10)
A[4]=1
print(A)

#7. Create a vector with values ranging from 10 to 49
A = np.arange(10,50)
print(A)

#8. Reverse a vector
A = A[::-1]
print(A)
A = np.flip(A)
print(A)

#9. Create a 3*3 matrix with values ranging from 0 to 8
A = np.arange(9).reshape(3,3)
print(A)

#10. Find indices of non-zero elements from [1,2,0,0,4,0]
A = np.nonzero([1,2,0,0,4,0]) #zero가 아닌 element의 index를 return
print(A)

#11. Create a 3*3 identity matrix
A = np.eye(3)
print(A)

#12. Create a 3*3*3 array with random values 
A = np.random.random((3,3,3)) #numpy의 random module의 random function
print(A)

#13. Create a 10*10 array with random values and find the minimum and maximum values
A = np.random.random((10,10))
print(A.min(), A.max())

#14. Create a random vector of size 30 and find the mean value
A = np.random.random(30)
print(A.mean())

#15. Create a 2d array with 1 on the border and 0 inside
A = np.ones((10,10))
A[1:-1, 1:-1] = 0 
print(A) #아직 모르겠음

#16. How to add a border (filled with 0's) around an existing array?
A = np.ones((5,5))
A = np.pad(A, pad_width=1, mode = 'constant', constant_values=0)
print(A)

#17. What is the result of the following expression?
print(0*np.nan) #nan은 정의할 수 없는 숫자
print(np.nan == np.nan)
print(np.inf > np.nan) #inf는 무한대
print(np.nan - np.nan)
print(np.nan in set([np.nan]))
print(0.3 == 3*0.1) # 3*0.1= 0.30000...4 라서 false

#18. Create a 5*5 matrix with values 1,2,3,4 just below the diagonal
A = np.diag(1+np.arange(4), k=-1) # k=-1하면 대각선 아래에 적용됨
print(A)

#19. Create a 8*8 matrix and fill it with a checkerboard pattern
A = np.zeros((8,8))
A[1::2,::2] = 1
A[::2,1::2] = 1
print(A)

#20. Consider a (6,7,8) shape array, what is the index (x,y,z) of the 100th element?
A = np.unravel_index(100,(6,7,8))
print(A) # 잘 모르겠음

#21. Create a checkerboard 8*8 matrix using the tile function
A = np.tile(np.array([[0,1],[1,0]]), (4,4)) #([0,1],[1,0])를 가로 4번, 세로 4번 반복
print(A)

#22. Normalize a 5*5 random matrix
A = np.random.random((5,5))
print(A)
A = ((A - A.mean()) / A.std()) # 정규화 공식
print(A)

#23. Create a custom dtype that describes a color as four unsigned bytes
color = np.dtype([("r", np.ubyte, 1),
                  ("g", np.ubyte, 1),
                  ("b", np.ubyte, 1),
                  ("a", np.ubyte, 1)]) # 무슨말일까

#24. Multiply a 5*3 matrix by a 3*2 matrix (real matrix product)
A = np.dot(np.ones((5,3)), np.ones((3,2)))
print(A) # 5*2 행렬

#25. Given a 1D array, negate all elements which are between 3 and 8, in place
A = np.arange(11)
print(A)
A[(3<A)&(A<8)] *= -1 #원소에 -1 곱하기
print(A)

#26 What is the output of the following script?
print(sum(range(5),-1)) # 9
from numpy import *
print(sum(range(5),-1)) # 10 뭐때문에 다를까,,

#27. Consider an integer vector Z, which of these expressions are legal?
"""
Z**Z
2 << Z >> 2
Z <- Z
1j*Z
Z/1/1
Z<Z>Z
"""
Z = np.arange(10)
print(Z**Z)
print(2<<Z>>2)
print( Z<-Z)
print(1j*Z)
print(Z/1/1)
#print(Z<Z>Z) #얘만 오류뜸 왜??

#28. What are the result of the following expressions?
print(np.array(0) / np.array(0)) #array 나누기 불가능
print(np.array(0) // np.array(0)) # //하면 소수점 이하 버리고 정수만 구하기
print(np.array([np.nan]).astype(int).astype(float))

#29. How to round away from zero a float array ?
A = np.random.uniform(-10,10,10) # -10에서 10까지 10개
print(A)
print(np.copysign(np.ceil(np.abs(A)),A)) #절대값 씌우고, ceil(올림)하고 기존 A랑 부호 똑같이 copysign

#30. How to find common values between two arrays?
A1 = np.random.randint(0,10,10)
A2 = np.random.randint(0,10,10)
print(A1, A2)
print(np.intersect1d(A1,A2)) #순서 상관없이 공통인자

#31. How to ignore all numpy warnings (not recommended)?
defaults = np.seterr(all="ignore")
A = np.ones(1) /0 # test문장

#32. Is the following expressions true?
print(np.sqrt(-1) == np.emath.sqrt(-1)) #false 왼쪽은 nan, 오른쪽은 1j

#33. How to get the dates of yesterday, today and tomorrow?
yesterday = np.datetime64('today') - np.timedelta64(1)
today = np.datetime64('today')
tomorrow = np.datetime64('today') +np.timedelta64(1) # 기본값이 day 
print(yesterday)
print(today)
print(tomorrow)

#34. How to get all the dates corresponding to the month of July 2016?
A = np.arange('2016-07', '2016-08', dtype ='datetime64[D]')
print(A)

#35. How to compute ((A+B)*(-A/2)) in place (without copy)?
A = np.ones(3)*1
B = np.ones(3)*2
C = np.ones(3)*3
np.add(A,B,out=B)
np.divide(A,2,out=A)
np.negative(A,out=A)
np.multiply(A,B,out=A)
print(A)

#36. Extract the integer part of a random array of positive numbers using 4 different methods
A = np.random.uniform(0,10,10)

print(A - A%1)
print(A // 1)
print(np.floor(A))
print(A.astype(int))
print(np.trunc(A))

#37. Create a 5x5 matrix with row values ranging from 0 to 4
A = np.zeros((5,5))
A += np.arange(5)
print(A)

#38. Consider a generator function that generates 10 integers and use it to build an array
def generate():
    for x in range(10):
        yield x # return대신 yield 사용 => 여러번에 걸쳐 입출력을 받을 수 있음
        
A = np.fromiter(generate(), dtype=float, count=-1)
print(A)

#39. Create a vector of size 10 with values ranging from 0 to 1, both excluded
A = np.linspace(0,1,11,endpoint=False)[1:] # endpoint는 1이안되게, [1:]로 인덱스 0을 제외
print(A)

#40. Create a random vector of size 10 and sort it
A = np.random.random(10)
A.sort()
print(A)

#41. How to sum a small array faster than np.sum?
A = np.arange(10)
print(np.add.reduce(A)) # add.reduce는 sum과 등가

#42. Consider two random array A and B, check if they are equal
A = np.random.randint(0,2,5)
B = np.random.randint(0,2,5)

equal = np.array_equal(A,B)
print(equal)

#43. Make an array immutable (read-only)
A = np.zeros(10)
A.flags.writeable = False
#A[0] = 1    => ValueError

#44. Consider a random 10x2 matrix representing cartesian coordinates, convert them to polar coordinates
A = np.random.random((10,2))
X,Y = A[:,0], A[:,1]
print(X)
R = np.sqrt(X**2+Y**2)  #극좌표 길이
T = np.arctan2(Y,X) #극좌표 세타
print(R)
print(T)

#45. Create random vector of size 10 and replace the maximum value by 0
A = np.random.random(10)
A[A.argmax()] = 0
print(A)

#46. Create a structured array with x and y coordinates covering the [0,1]x[0,1] area
A = np.zeros((5,5), [('x',float),('y',float)]) 
A['x'], A['y'] = np.meshgrid(np.linspace(0,1,5),
                             np.linspace(0,1,5))
print(A)

#47. Given two arrays, X and Y, construct the Cauchy matrix C (Cij =1/(xi - yj))
X = np.arange(8)
Y = X + 0.5
C = 1.0 / np.subtract.outer(X,Y) # outer가 xi, yj 각각을 의미
print(C)

#48. Print the minimum and maximum representable value for each numpy scalar type
for dtype in [np.int8, np.int32, np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
   print(np.finfo(dtype).min)
   print(np.finfo(dtype).max)
   print(np.finfo(dtype).eps) #eps는 표현가능한 가장 작은 값

#49. How to print all the values of an array?
np.set_printoptions(threshold=float("inf")) #요약을 하는 요소의 수(threshold)
A = np.zeros((16,16))
print(A)


#50. How to find the closest value (to a given scalar) in a vector?
A = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(A-v).argmin()) # 절대값차이가 가장 작은게 closest value임
print(A[index])

#51. Create a structured array representing a position (x,y) and a color (r,g,b)
A = np.zeros(10, [ ('position', [('x', float, 1),
                                 ('y', float, 1)]),
                   ('color', [('r', float, 1),
                              ('g', float, 1),
                              ('b', float, 1)])])

print(A) # position과 color가 한 쌍으로 묶임

print('\n\n')
#52. Consider a random vector with shape (100,2) representing coordinates, find point by point distances
Z = np.random.random((10,2)) # 랜덤으로 벡터 생성
X,Y = np.atleast_2d(Z[:,0], Z[:,1]) # 0열과 1열을 각각 x,y에 할당
print(X)
print(X.T)
print('\n\n')
print(X-X.T)
print('\n\n')
D = np.sqrt( (X-X.T)**2 + (Y-Y.T)**2)
print(D)

#53. How to convert a float (32 bits) array into an integer (32 bits) in place?
Z = (np.random.rand(10)*100).astype(np.float32) #형변환해서 float으로
Y = Z.view(np.int32) #Z를 int32 타입으로 변환해서 복사(view)
print(Y)

#54. How to read the following file?
from io import StringIO

s = StringIO('''1, 2, 3, 4, 5

                6,  ,  , 7, 8

                 ,  , 9,10,11
''')

Z = np.genfromtxt(s, delimiter=",", dtype=np.int) # 콤마를 구분문자로 하고, int형으로 반환
print(Z)

#55. What is the equivalent of enumerate for numpy arrays? 
Z = np.arange(9).reshape(3,3) #3*3 배열 생성
for index, value in np.ndenumerate(Z): #ndenumerate 함수로 값을 하나하나 셀 수 있음
    print(index, value)

#56. Generate a generic 2D Gaussian-like array
X, Y = np.meshgrid(np.linspace(-1,1,10), np.linspace(-1,1,10))
D = np.sqrt(X*X+Y*Y)
sigma, mu = 1.0, 0.0
G = np.exp(-( (D-mu)**2 / ( 2.0 * sigma**2 ) ) )
#print(G) #솔직히 모르겠음

#57. How to randomly place p elements in a 2D array?
n = 10
p = 3
Z = np.zeros((n,n))
np.put(Z, np.random.choice(range(n*n), p, replace=False),1) #Z행렬에 3개의 요소를 랜덤하게 1로 변환
print(Z)

#58. Subtract the mean of each row of a matrix
X = np.random.rand(5, 10)
Y = X - X.mean(axis=1, keepdims=True) # 걍 빼면 됨

print(Y)

#59. How to sort an array by the nth column?
Z = np.random.randint(0,10,(3,3))
print(Z)
print(Z[Z[:,1].argsort()]) #1열의 크기순대로 인덱스가 반환되고 그에 따라 섞임

#60. How to tell if a given 2D array has null columns?
Z = np.random.randint(0,3,(3,10))
print((~Z.any(axis=0)).any()) # null인 column이 하나라도 있으면 true

#61. Find the nearest value from a given value in an array
Z = np.random.uniform(0,1,10)
print(Z)
z = 0.5
m = Z.flat[np.abs(Z - z).argmin()] # 차이의 절대값이 가장 작은것
print(m)

#62. Considering two arrays with shape (1,3) and (3,1), how to compute their sum using an iterator?
A = np.arange(3).reshape(3,1)
B = np.arange(3).reshape(1,3)
it = np.nditer([A,B,None]) # 이게뭐지
for x,y,z in it: 
    z[...] = x + y
print(it.operands[2]) #z를 가져오는것

#63. Create an array class that has a name attribute
class NamedArray(np.ndarray):
    def __new__(cls, array, name="no name"): 
        obj = np.asarray(array).view(cls)
        obj.name = name # name을 넣을수있음
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.info = getattr(obj, 'name', "no name") # obj에 존재하는 속성을 가져옴

Z = NamedArray(np.arange(10), "range_10")
print (Z.name)

#64. Consider a given vector, how to add 1 to each element indexed by a second vector (be careful with repeated indices)?
Z = np.ones(10)
I = np.random.randint(0,len(Z),20) # 0에서 10까지 20개 랜덤으로 
Z += np.bincount(I, minlength=len(Z)) # 0에서 10까지 빈도수를 세니까 원소수 10개
print(Z)

#65. How to accumulate elements of a vector (X) to an array (F) based on an index list (I)? 
X = [1,2,3,4,5,6] 
I = [3,4,5,6,7,8] 
F = np.bincount(I,X) # [0 0 0 1 2 3 4 5 6] 
print(F)

#66. Considering a (w,h,3) image of (dtype=ubyte), compute the number of unique colors
w,h = 16,16
I = np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
F = I[...,0]*256*256 + I[...,1]*256 +I[...,2]
print(len(np.unique(F)))



#67. Considering a four dimensions array, how to get sum over the last two axis at once?
A = np.random.randint(0,10,(3,4,3,4)) #0에서 10까지 3*4*3*4 개
sum = A.sum(axis=(-2,-1)) # last two axis 더하는법
print(sum)

#68. Considering a one-dimensional vector D, how to compute means of subsets of D using a vector S of same size describing subset indices?
D = np.random.uniform(0,1,100)
S = np.random.randint(0,10,100)
D_sums = np.bincount(S, weights=D) 
D_counts = np.bincount(S)
D_means = D_sums / D_counts # bincount는 다시 이해하기.
print(D_means)

#69. How to get the diagonal of a dot product?
A = np.random.uniform(0,1,(5,5))
B = np.random.uniform(0,1,(5,5))

print(np.diag(np.dot(A, B))) #내적해도 5*5행렬이므로

#70. Consider the vector [1, 2, 3, 4, 5], how to build a new vector with 3 consecutive zeros interleaved between each value?
Z = np.array([1,2,3,4,5])
nz = 3 # 간격이 3 
Z0 = np.zeros(len(Z) + (len(Z)-1)*(nz)) #0 0 0 이 들어가는 부분 포함해서 길게 형성
Z0[::nz+1] = Z ##nz+1 간격으로 Z를 Z0에 삽입
print(Z0)

#71. Consider an array of dimension (5,5,3), how to mulitply it by an array with dimensions (5,5)?
A = np.ones((5,5,3))
B = 2*np.ones((5,5))
print(A * B[:,:,None]) # 처음 5*5는 곱하고 마지막 차원에 대해서는 곱을 안함

#72. How to swap two rows of an array?
A = np.arange(25).reshape(5,5)
A[[0,1]] = A[[1,0]] # A[[0,1]] == A[[0,1],:]를 의미 => 0행과 1행을 swap
print(A) 

#73. Consider a set of 10 triplets describing 10 triangles (with shared vertices), 
# find the set of unique line segments composing all the triangles
faces = np.random.randint(0,100,(10,3)) # 문제 이해가 어렵당.
F = np.roll(faces.repeat(2,axis=1),-1,axis=1)
F = F.reshape(len(F)*3,2)
F = np.sort(F,axis=1)
G = F.view( dtype=[('p0',F.dtype),('p1',F.dtype)] )
G = np.unique(G)
print(G)

#74. Given an array C that is a bincount, how to produce an array A such that np.bincount(A) == C?
C = np.bincount([1,1,2,3,4,4,6]) # [0 2 1 1 2 0 1]
A = np.repeat(np.arange(len(C)), C) # [0 1 2 3 4 5 6] => [0 1 1 2 3 4 4 0 6] 이므로 bincount값 같음
print(A)

#75. How to compute averages using a sliding window over an array?
def moving_average(a, n=3) : #sliding window => 일정범위를 가진 것을 유지하면서 이동하는 것
    ret = np.cumsum(a, dtype=float) # 누적 합 계산 [0 1 3 6 10 15 21 28 36 45]
    ret[n:] = ret[n:] - ret[:-n] # [-22 -11 0 11 22 33 44]
    return ret[n - 1:] / n #[0 11/3 22/3 11 44/3]
Z = np.arange(10) # [0 1 2 3 4 5 6 7 8 9]
print(moving_average(Z, n=3))

#76. Consider a one-dimensional array Z, build a two-dimensional array 
# whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 
# (last row should be (Z[-3],Z[-2],Z[-1])
from numpy.lib import stride_tricks

def rolling(a, window): # window 범위 크기의 sub를 추출할 수 있음
    shape = (a.size - window + 1, window)
    strides = (a.itemsize, a.itemsize)
    return stride_tricks.as_strided(a, shape=shape, strides=strides)
Z = rolling(np.arange(10), 3) 
print(Z)

#77. How to negate a boolean, or to change the sign of a float inplace?
Z = np.random.randint(0,2,100)
np.logical_not(Z, out=Z) # Z의 원소에 모두 NOT 적용

Z = np.random.uniform(-1.0,1.0,100)
np.negative(Z, out=Z) # 부호 변경

#78. Consider 2 sets of points P0,P1 describing lines (2d) and a point p,
# how to compute distance from p to each line i (P0[i],P1[i])?
def distance(P0, P1, p):
    T = P1 - P0
    L = (T**2).sum(axis=1) # row에 대해 모두 합
    U = -((P0[:,0]-p[...,0])*T[:,0] + (P0[:,1]-p[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = P0 + U*T - p
    return np.sqrt((D**2).sum(axis=1)) # 수식 아직 이해 안됨.

P0 = np.random.uniform(-10,10,(10,2))
P1 = np.random.uniform(-10,10,(10,2))
p  = np.random.uniform(-10,10,( 1,2))
print(distance(P0, P1, p))

#79. Consider 2 sets of points P0,P1 describing lines (2d) and a set of points P, 
# how to compute distance from each point j (P[j]) to each line i (P0[i],P1[i])?
P0 = np.random.uniform(-10, 10, (10,2))
P1 = np.random.uniform(-10,10,(10,2))
p = np.random.uniform(-10, 10, (10,2))
print(np.array([distance(P0,P1,p_i) for p_i in p])) # 78번과 마찬가지

#80. Consider an arbitrary array, write a function that extract a subpart 
# with a fixed shape and centered on a given element (pad with a fill value when necessary)
Z = np.random.randint(0,10,(10,10))
shape = (5,5)
fill  = 0
position = (1,1)

R = np.ones(shape, dtype=Z.dtype)*fill
P  = np.array(list(position)).astype(int)
Rs = np.array(list(R.shape)).astype(int)
Zs = np.array(list(Z.shape)).astype(int)

R_start = np.zeros((len(shape),)).astype(int)
R_stop  = np.array(list(shape)).astype(int)
Z_start = (P-Rs//2)
Z_stop  = (P+Rs//2)+Rs%2

R_start = (R_start - np.minimum(Z_start,0)).tolist()
Z_start = (np.maximum(Z_start,0)).tolist()
R_stop = np.maximum(R_start, (R_stop - np.maximum(Z_stop-Zs,0))).tolist()
Z_stop = (np.minimum(Z_stop,Zs)).tolist()

r = [slice(start,stop) for start,stop in zip(R_start,R_stop)]
z = [slice(start,stop) for start,stop in zip(Z_start,Z_stop)]
R[r] = Z[z]
print(Z)
print(R)

#81. Consider an array Z = [1,2,3,4,5,6,7,8,9,10,11,12,13,14], 
# how to generate an array R = [[1,2,3,4], [2,3,4,5], [3,4,5,6], ..., [11,12,13,14]]?
Z = np.arange(1,15,dtype=np.uint32) # 조건대로 배열 생성(4바이트)
R = stride_tricks.as_strided(Z,(11,4),(4,4)) # 11*4 형상
print(R)

#82. Compute a matrix rank
Z = np.random.uniform(0,1,(10,10))
U, S, V = np.linalg.svd(Z) # 특이값 분해 방법 SVD
rank = np.sum(S > 1e-10)
print(rank)

#83. How to find the most frequent value in an array?
Z = np.random.randint(0,10,50)
print(np.bincount(Z).argmax()) # 빈도수를 카운트해서 가장 큰 값을 고르면 됨.

#84. Extract all the contiguous 3x3 blocks from a random 10x10 matrix
Z = np.random.randint(0,5,(10,10))
n = 3 
i = 1 + (Z.shape[0]-3)
j = 1 + (Z.shape[1]-3)
C = stride_tricks.as_strided(Z, shape=(i, j, n, n), strides=Z.strides + Z.strides)
print(C) # Stride_tricks 사용법에 대한 정보가 많이 없어서 해석하기가 어려움.. 책을 한번 빌려봐야 될 것 같음.

#85. Create a 2D array subclass such that Z[i,j] == Z[j,i]
class Symetric(np.ndarray):
    def __setitem__(self, index, value):
        i,j = index
        super(Symetric, self).__setitem__((i,j), value)
        super(Symetric, self).__setitem__((j,i), value) # 값 교환

    def symetric(Z):
        return np.asarray(Z + Z.T - np.diag(Z.diagonal())).view(Symetric) # 사실 잘 모르겠다.

S = symetric(np.random.randint(0,10,(5,5)))
S[2,3] = 42
print(S)

#86. Consider a set of p matrices with shape (n,n) and a set of p vectors with shape (n,1). 
# How to compute the sum of the p matrix products at once? (result has shape (n,1))
p, n = 10, 20
M = np.ones((p,n,n)) # 10, 20, 20
V = np.ones((p,n,1)) # 10, 20, 1
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]]) # 텐서곱, 지정된 축에 따라 내적 수행
print(S)

#87. Consider a 16x16 array, how to get the block-sum (block size is 4x4)?
Z = np.ones((16,16))
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0), # 0부터 15까지 4개
                                       np.arange(0, Z.shape[1], k), axis=1)
print(S) # 4*4

#88. How to implement the Game of Life using numpy arrays? 
def iterate(Z): # game of life..?
    # Count neighbours
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2]                + Z[1:-1,2:] +
         Z[2:  ,0:-2] + Z[2:  ,1:-1] + Z[2:  ,2:])

    # Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z

Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)
print(Z)

#89. How to get the n largest values of an array
Z = np.arange(10000)
np.random.shuffle(Z)
n = 5

print (Z[np.argsort(Z)[-n:]]) # 오름차순 소팅후에 마지막 n개 추출 

#90. Given an arbitrary number of vectors, 
# build the cartesian product (every combinations of every item) 
def cartesian(arrays):
    arrays = [np.asarray(a) for a in arrays] # 복사..는 아니고 참조
    shape = (len(x) for x in arrays)

    ix = np.indices(shape, dtype=int)
    ix = ix.reshape(len(arrays), -1).T

    for n, arr in enumerate(arrays):
        ix[:, n] = arrays[n][ix[:, n]]

    return ix

print (cartesian(([1, 2, 3], [4, 5], [6, 7])))

#91. How to create a record array from a regular array?
Z = np.array([("Hello", 2.5, 3),
              ("World", 3.6, 2)])
R = np.core.records.fromarrays(Z.T,
                               names='col1, col2, col3',
                               formats = 'S8, f8, i8')
print(R)

#92. Consider a large vector Z, compute Z to the power of 3 using 3 different methods 
x = np.random.rand(int(5e7))

#%timeit np.power(x,3)
#%timeit x*x*x
#%timeit np.einsum('i,i,i->i',x,x,x)

#93. Consider two arrays A and B of shape (8,3) and (2,2). 
#How to find rows of A that contain elements of each row of B regardless of the order of the elements in B? 
A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))

C = (A[..., np.newaxis, np.newaxis] == B) # 차원을 바꾸는 함수 (B와 같이) 
rows = np.where(C.any((3,1)).all(1))[0]
print(rows)

#94. Considering a 10x3 matrix, extract rows with unequal values (e.g. [2,2,3])
Z = np.random.randint(0,5,(10,3))
print(Z)

E = np.all(Z[:,1:] == Z[:,:-1], axis=1)
U = Z[~E]
print(U)

#95. Convert a vector of ints into a matrix binary representation 
I = np.array([0, 1, 2, 3, 15, 16, 32, 64, 128])
B = ((I.reshape(-1,1) & (2**np.arange(8))) != 0).astype(int) # 이진법을 수식으로
print(B[:,::-1])

#96. Given a two dimensional array, how to extract unique rows?
Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

#97. Considering 2 vectors A & B, write the einsum equivalent of inner, outer, sum, and mul function 
A = np.random.uniform(0,1,10)
B = np.random.uniform(0,1,10)

np.einsum('i->', A)       # np.sum(A)
np.einsum('i,i->i', A, B) # A * B
np.einsum('i,i', A, B)    # np.inner(A, B) 내적
np.einsum('i,j->ij', A, B)    # np.outer(A, B) 외적

#98. Considering a path described by two vectors (X,Y), how to sample it using equidistant samples?
phi = np.arange(0, 10*np.pi, 0.1)
a = 1
x = a*phi*np.cos(phi)
y = a*phi*np.sin(phi)

dr = (np.diff(x)**2 + np.diff(y)**2)**.5 # segment lengths
r = np.zeros_like(x)
r[1:] = np.cumsum(dr)                # integrate path
r_int = np.linspace(0, r.max(), 200) # regular spaced path
x_int = np.interp(r_int, r, x)       # integrate path
y_int = np.interp(r_int, r, y) #수식적 이해 부족

#99. Given an integer n and a 2D array X, 
# select from X the rows which can be interpreted as draws from a multinomial distribution with n degrees, 
# i.e., the rows which only contain integers and which sum to n.
X = np.asarray([[1.0, 0.0, 3.0, 8.0],
                [2.0, 0.0, 1.0, 1.0],
                [1.5, 2.5, 1.0, 0.0]])
n = 4
M = np.logical_and.reduce(np.mod(X, 1) == 0, axis=-1)
M &= (X.sum(axis=-1) == n)
print(X[M])

#100. Compute bootstrapped 95% confidence intervals for the mean of a 1D array X 
# (i.e., resample the elements of an array with replacement N times, 
# compute the mean of each sample, and then compute percentiles over the means).
X = np.random.randn(100) # random 1D array
N = 1000 # number of bootstrap samples
idx = np.random.randint(0, X.size, (N, X.size))
means = X[idx].mean(axis=1)
confint = np.percentile(means, [2.5, 97.5]) # 백분위수 구하기
print(confint)