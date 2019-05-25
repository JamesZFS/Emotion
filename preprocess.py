import os
import re

import numpy as np
import pickle
from tqdm import tqdm

tag_list = ('感动', '同情', '无聊', '愤怒', '搞笑', '难过', '新奇', '温馨')

tag_pattern = \
	re.compile('感动:(\d+) 同情:(\d+) 无聊:(\d+) 愤怒:(\d+) 搞笑:(\d+) 难过:(\d+) 新奇:(\d+) 温馨:(\d+)', re.U)

embedding_dict = None
vec_dim = None


def get_tag(line: str):
	'''

	:param line: one line of article
	:return: tag (normalized) , ndarray
	'''
	mt = tag_pattern.search(line)
	assert mt
	tag = list(map(int, mt.groups()))
	sum_up = sum(tag)
	tag = list(map(lambda x: x / sum_up, tag))  # normalize

	return np.array(tag)


def get_embedding_list(line: str, time_steps: int = 0, silent: bool = False):
	'''

	:param line: one line of article
	:return: an ndarray of word embeddings in the article (time_step, vec_dim)
	'''
	global embedding_dict, vec_dim
	if embedding_dict is None:
		print('loading embedding_dict now...')
		with open('embeddings/embeddings.dict', 'rb') as f:
			embedding_dict = pickle.load(f)
		vec_dim = len(embedding_dict['，'])
		print('\rembedding_dict loaded, vec_dim = %d' % vec_dim)

	mt = tag_pattern.search(line)
	assert mt
	line = line[mt.end():].strip()
	words = line.split(' ')
	embedding_list = []
	unknowns = set()
	for word in words:
		if not embedding_dict.get(word):
			embedding_list.append(np.random.randn(vec_dim))  # unk
			unknowns.add(word)
		else:
			embedding_list.append(embedding_dict[word])
	if silent == False: print('unknown words:', unknowns)
	if time_steps > 0:
		padding = time_steps - len(embedding_list)
		if padding < 0:
			# article too long, randomly slice short
			cut = np.random.randint(0, len(embedding_list) - time_steps)
			embedding_list = embedding_list[cut: cut + time_steps]
		else:
			# pad zeros to time_step
			for _ in range(padding):
				embedding_list.append(np.zeros(vec_dim))

	return np.array(embedding_list)


def get_article_max_len(file: str):
	'''
	stat longest article in a dataset

	:param file: filename of dataset
	:return: int, max_len
	'''
	with open(file, 'r') as f:
		max_len = 0
		while True:
			line = f.readline()
			if not line: break
			mt = tag_pattern.search(line)
			assert mt
			line = line[mt.end():].strip()
			words = line.split(' ')
			max_len = max(max_len, len(words))

	return max_len


def generator_from_file(raw_path: str, batch_size: int = 10):
	'''
	generate data from file

	:param raw_path: news path
	:return: yield (X, Y) X shape like (batch_size, 500, 300), Y shape like (batch_size, 8)
	'''
	with open(raw_path, 'r') as f:
		while True:
			X, Y = [], []
			for _ in range(batch_size):
				line = f.readline()
				if not line:
					f.seek(0)  # start over
					line = f.readline()
				Y.append(get_tag(line))
				X.append(get_embedding_list(line, time_steps=500, silent=True))

			yield np.array(X), np.array(Y)


def generator_from_file_debug(raw_path: str, batch_size: int = 10):
	'''
	generate data from file

	:param raw_path: news path
	:return: yield (X, Y, articles) X shape like (batch_size, 500, 300), Y shape like (batch_size, 8),
		articles shape like (batch_size,) of str
	'''
	with open(raw_path, 'r') as f:
		while True:
			X, Y, articles = [], [], []
			for _ in range(batch_size):
				line = f.readline()
				if not line:
					f.seek(0)  # start over
					line = f.readline()
				Y.append(get_tag(line))
				X.append(get_embedding_list(line, time_steps=500, silent=True))
				mt = tag_pattern.search(line)
				articles.append(''.join(line[mt.end():].split(' ')))

			yield np.array(X), np.array(Y), articles


if __name__ == '__main__':
	# mode = 'train'
	# with open('data/sina/sinanews.%s' % mode, 'r') as f:
	# 	lines = f.readlines()
	#
	# X = []
	# Y = []
	# for line in tqdm(lines, desc='converting...'):
	# 	Y.append(get_tag(line))
	# 	X.append(get_embedding_list(line))
	#
	# print('saving now...')
	# np.save('data/done/X.%s.npy' % mode, X)
	# np.save('data/done/Y.%s.npy' % mode, Y)

	print(get_article_max_len('data/sina/sinanews.demo'))
