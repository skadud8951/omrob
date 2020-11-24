# coding: utf-8
import sys
sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb


corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
corpus_size = len(corpus)

model = RnnlmGen()
model.load_params('../dataset/Rnnlm.pkl')

# start 문자와 skip 문자 설정
start_word = 'you' #you를 시작단어로 설정
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$'] # 샘플링 하지않을 단어 목록
skip_ids = [word_to_id[w] for w in skip_words]
# 문장 생성
word_ids = model.generate(start_id, skip_ids) #생성, word_ids는 ID로 이루어져있으므로 변환해야됨
txt = ' '.join([id_to_word[i] for i in word_ids]) #ID를 word로 바꿔서 공백을 주어 연결 
txt = txt.replace(' <eos>', '.\n')
print(txt)
