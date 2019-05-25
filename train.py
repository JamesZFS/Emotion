import os

import numpy as np
from keras import Sequential, layers, optimizers

from preprocess import generator_from_file

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac


def build_RNN():
	'''
	build and compile an RNN models

	:return: RNN models
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


if __name__ == '__main__':
	model = build_RNN()

	data_gen = generator_from_file('data/sina/sinanews.train', batch_size=10)  # around 2000 articles in total
	model.fit_generator(data_gen, epochs=10, steps_per_epoch=10)

	test_gen = generator_from_file('data/sina/sinanews.test', batch_size=10)
	train_loss, train_acc = model.evaluate_generator(data_gen, steps=20)
	print(train_loss, train_acc)

	model.save('models/state2.0.h5')
