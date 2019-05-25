import os
import numpy as np

from keras import Sequential, layers, optimizers

import preprocess

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac


def build_RNN():
	'''
	build and compile an RNN model

	:return: RNN model
	'''
	dims = (300, 100, 8)
	model = Sequential()
	model.add(layers.InputLayer(input_shape=(None, dims[0])))
	model.add(layers.Masking(mask_value=0.0))
	model.add(layers.GRU(units=dims[1], return_sequences=False))
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(units=dims[2], activation='softmax'))

	opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	model.compile(opt, loss='mse', metrics=['acc'])
	model.summary()

	return model


def load_dataset(mode: str):
	'''

	:param mode: 'demo' / 'train' / 'test'
	:return: X, Y
	'''
	X = np.load('data/done/X.%s.npy' % mode)
	Y = np.load('data/done/Y.%s.npy' % mode)
	assert len(X) == len(Y)

	return X, Y


def generator_from_file(raw_path: str, batch_size: int = 10):
	'''
	generate data from file

	:param raw_path: news path
	:return: yield (x, y) x shape like (xxx, 300), y shape like (8,)
	'''
	with open(raw_path, 'r') as f:
		while True:
			X, Y = [], []
			for _ in range(batch_size):
				line = f.readline()
				if not line:
					f.seek(0) # start over
					line = f.readline()
				Y.append(preprocess.get_tag(line))
				X.append(preprocess.get_embedding_list(line, time_steps=500, silent=True))

			yield np.array(X), np.array(Y)


if __name__ == '__main__':
	model = build_RNN()

	data_gen = generator_from_file('data/sina/sinanews.train', batch_size=10)  # around 2000 articles in total
	model.fit_generator(data_gen, epochs=10, steps_per_epoch=10)

	test_gen = generator_from_file('data/sina/sinanews.test', batch_size=10)
	train_loss, train_acc = model.evaluate_generator(data_gen, steps=20)
	print(train_loss, train_acc)

	model.save('model/state.h5')
