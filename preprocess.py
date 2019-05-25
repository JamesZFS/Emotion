import os
import pickle
import re

import numpy as np
from keras.preprocessing.text import Tokenizer

from encoder import tag_pattern

print('now loading encoder and embedding matrix...')
with open('embeddings/encoder.tknz', 'rb') as f:
	tknz = pickle.load(f)
assert isinstance(tknz, Tokenizer)
word_index = tknz.word_index
embedding_matrix = np.load('embeddings/embedding_matrix.npy')
vocab_size = embedding_matrix.shape[0]
vec_dim = embedding_matrix.shape[1]
print('loaded, vocab_size = %d, vec_dim = %d' % (vocab_size, vec_dim))
word_pattern = re.compile('\D+', re.U)


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


def get_encoded_text(line: str, article_length: int = 500, silent: bool = True):
	'''get encoded text of an article

	:param line: one line of article
	:return: array of word indexes
	'''
	mt = tag_pattern.search(line)
	assert mt
	line = line[mt.end():].strip()
	# get words
	words = filter_words(line.split(' '))
	encoded_list = []
	# encoding
	for word in words:
		index = word_index.get(word)
		if index is None:
			if not silent: print('word %s is unknown' % word)
		else:
			encoded_list.append(index)
	# padding
	if article_length > 0:
		padding = article_length - len(encoded_list)
		if padding < 0:
			# article too long, randomly slice short
			cut = np.random.randint(0, len(encoded_list) - article_length)
			encoded_list = encoded_list[cut: cut + article_length]
		else:
			# pad zeros to time_step
			for _ in range(padding):
				encoded_list.append(0)

	return encoded_list


def filter_words(words: list):
	filtered = []
	for word in words:
		filtered += word_pattern.findall(word)
	return filtered


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


def load_dataset_from_file(raw_path: str, shuffle: bool = True):
	'''load dataset from file

	:param raw_path: news path
	:return: (X, Y) where X shape like (n_article, 500), Y shape like (n_article, 8)
	'''
	with open(raw_path, 'r') as f:
		lines = f.readlines()
	if shuffle: np.random.shuffle(lines)

	X, Y = [], []
	for line in lines:
		X.append(get_encoded_text(line))
		Y.append(get_tag(line))

	return np.array(X), np.array(Y)


def generator_from_file(raw_path: str, batch_size: int = 10, shuffle: bool = True):
	'''
	generate data from file

	:param raw_path: news path
	:return: yield (X, Y) X shape like (batch_size, 500), Y shape like (batch_size, 8)
	'''
	with open(raw_path, 'r') as f:
		lines = f.readlines()
	if shuffle: np.random.shuffle(lines)

	cur_idx = 0
	while True:
		X, Y = [], []
		for _ in range(batch_size):
			line = lines[cur_idx]
			cur_idx += 1
			if cur_idx == len(lines): cur_idx = 0  # start over
			Y.append(get_tag(line))
			X.append(get_encoded_text(line))

		yield np.array(X), np.array(Y)


def generator_from_file_debug(raw_path: str, batch_size: int = 10, shuffle: bool = True):
	'''
	generate data from file

	:param raw_path: news path
	:return: yield (X, Y, articles) X shape like (batch_size, 500), Y shape like (batch_size, 8),
		articles shape like (batch_size,) of str
	'''
	with open(raw_path, 'r') as f:
		lines = f.readlines()
	if shuffle: np.random.shuffle(lines)

	cur_idx = 0
	while True:
		X, Y, articles = [], [], []
		for _ in range(batch_size):
			line = lines[cur_idx]
			cur_idx += 1
			if cur_idx == len(lines): cur_idx = 0  # start over
			Y.append(get_tag(line))
			X.append(get_encoded_text(line))
			mt = tag_pattern.search(line)
			articles.append(''.join(line[mt.end():].split(' ')))

		yield np.array(X), np.array(Y), articles


def split_train_val_from_file(raw_path: str, output_dir: str, val_size: float = 0.1, shuffle: bool = True):
	'''
	split train and val data, dump to output_dir

	:param raw_path: news path
	'''
	with open(raw_path, 'r') as f:
		lines = f.readlines()
	if shuffle: np.random.shuffle(lines)

	cut = int(len(lines) * val_size)
	train = lines[cut:]
	val = lines[:cut]
	if not os.path.exists(output_dir): os.mkdir(output_dir)
	with open(os.path.join(output_dir, 'train'), 'w') as f:
		f.writelines(train)
	with open(os.path.join(output_dir, 'val'), 'w') as f:
		f.writelines(val)


if __name__ == '__main__':
	# split_train_val_from_file('data/sina/sinanews.train', 'data/train_val_test', val_size=0.15)
	# f = open('data/train_val_test/val', 'r')
	# lines = f.readlines()
	# print(len(lines))
	# print(lines[-1])
	# assert filter_words(['123', '123', '7月8号', '1天后']) == ['月', '号', '天后']
	data_gen = generator_from_file('data/sina/sinanews.demo', 1)
	X, Y = next(data_gen)
	print(X)
	print(Y)
