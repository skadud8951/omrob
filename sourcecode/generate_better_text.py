# coding: utf-8
import sys
sys.path.append('..')
from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)
#ptb에 들어있는 data를 적용

model = BetterRnnlmGen() #BetterRnnlmGen을 모델로 사용
model.load_params('../dataset/BetterRnnlm.pkl') #학습된 가중치가 저장된 pkl파일 이용

# start 문자와 skip 문자 설정
start_word = 'you' #시작단어를 you로 설정
start_id = word_to_id[start_word] # word를 ID로 바꾸어 start_id에 저장
skip_words = ['N', '<unk>', '$'] # 샘플링 하지않을 단어 설정
skip_ids = [word_to_id[w] for w in skip_words] #샘플링 하지않을 단어를 ID로 변환하여 저장
# 문장 생성
word_ids = model.generate(start_id, skip_ids) #모델을 통해 word_ids 리스트에 순서대로 저장
txt = ' '.join([id_to_word[i] for i in word_ids]) #id를 word로 변환한 뒤 공백을 두고 txt에 저장
txt = txt.replace(' <eos>', '.\n')

print(txt) 


model.reset_state() #모델을 reset

start_words = 'the meaning of life is' #시작단어를 바꿔서 한번 더 실행
start_ids = [word_to_id[w] for w in start_words.split(' ')] 
# 시작단어의 공백을 기준으로 나눠서 id로 바꾼 뒤 start_ids에 저장

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1) #predict를 위해 2차원배열로 바꾸는 과정
    model.predict(x) #이 과정은 왜 거치는지?

word_ids = model.generate(start_ids[-1], skip_ids)
# start_ids[-1]이 is에 대한 id값이므로 is를 첫 단어로 generate 하겠다는 뜻
word_ids = start_ids[:-1] + word_ids # 기존의 the meaning of life is 를 덧붙여서 전체 word 완성
txt = ' '.join([id_to_word[i] for i in word_ids]) #id를 word로 변환한 뒤 txt에 저장
txt = txt.replace(' <eos>', '.\n')
print('-' * 50) #구분선
print(txt)
