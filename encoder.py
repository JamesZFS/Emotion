from keras.preprocessing.text import Tokenizer
import pickle
import re

tag_list = ('感动', '同情', '无聊', '愤怒', '搞笑', '难过', '新奇', '温馨')

tag_pattern = \
	re.compile('感动:(\d+) 同情:(\d+) 无聊:(\d+) 愤怒:(\d+) 搞笑:(\d+) 难过:(\d+) 新奇:(\d+) 温馨:(\d+)', re.U)


def load_doc(raw_path: str):
	docs = []
	with open(raw_path, 'r') as f:
		lines = f.readlines()
		for line in lines:
			mt = tag_pattern.search(line)
			docs.append(line[mt.end() + 1:].strip())

	return docs


def get_encoder():
	'''get word encoder which maps a word that appears in corpus to an integer

	:return a Tokenizer
	'''
	docs = load_doc('data/sina/sinanews.demo') + \
		   load_doc('data/sina/sinanews.test') + \
		   load_doc('data/sina/sinanews.train')

	t = Tokenizer()
	t.filters += '0123456789'
	t.fit_on_texts(docs)

	return t


if __name__ == '__main__':
	import numpy as np

	# t = get_encoder()
	# with open('embeddings/encoder.tknz', 'wb') as f:
	# 	pickle.dump(t, f)
	with open('embeddings/encoder.tknz', 'rb') as f:
		t = pickle.load(f)
	assert isinstance(t, Tokenizer)

	print('loading embedding_dict now...')
	with open('embeddings/embeddings.dict', 'rb') as f:
		embedding_dict = pickle.load(f)
	assert isinstance(embedding_dict, dict)
	vec_dim = len(embedding_dict['，'])
	print('\rembedding_dict loaded, vec_dim = %d' % vec_dim)

	# turn dict into matrix
	vocab_size = len(t.word_index) + 1
	embedding_matrix = np.random.randn(vocab_size, vec_dim)
	for word, index in t.word_index.items():
		vec = embedding_dict.get(word)
		if vec is not None:
			embedding_matrix[index] = vec  # else random

	np.save('embeddings/embedding_matrix.npy', embedding_matrix)
