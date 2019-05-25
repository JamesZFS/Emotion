import os

import numpy as np
from keras import Sequential, layers, optimizers, callbacks

from preprocess import generator_from_file, split_train_val_from_file
from evaluate import evaluate_model

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac
version_name = 'model2.4' # todo

def build_RNN():
	'''build and compile an RNN models

	:return: RNN models
	'''
	dims = (300, 200, 8)
	model = Sequential()
	# embedding
	layers.Embedding(weights=[], trainable=False)
	model.add(layers.InputLayer(input_shape=(None, dims[0])))
	model.add(layers.Masking(mask_value=0.0))
	model.add(layers.GRU(units=dims[1], return_sequences=False))
	model.add(layers.Dropout(0.2))
	model.add(layers.Dense(units=dims[2], activation='softmax'))

	opt = optimizers.RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.01)
	model.compile(opt, loss='mse', metrics=['acc'])
	model.summary()

	return model


def build_CNN():
	pass


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
	# prepare data
	split_train_val_from_file('data/sina/sinanews.train', output_dir='data/runtime', val_size=0.15)
	train_gen = generator_from_file('data/runtime/train', batch_size=10)  # around 1700 articles
	val_gen = generator_from_file('data/runtime/val', batch_size=10)  # around 350 articles
	test_gen = generator_from_file('data/runtime/test', batch_size=10)  # around 2000 articles

	# build model
	model = build_RNN()
	# model.load_weights('models/model2.0 - best.h5')

	csv_logger = callbacks.CSVLogger(
		'logs/%s.csv' % version_name,
		separator=',', append=False)
	checkpoint_logger = callbacks.ModelCheckpoint(
		'models/%s - best.h5' % version_name,
		monitor='val_loss', verbose=1, save_best_only=True)

	# train
	model.fit_generator(train_gen, epochs=5, steps_per_epoch=100, # 100 x 10 = 1000 articles / epoch
						validation_data=val_gen, validation_steps=10, # 10 x 10 = 100
						verbose=1, callbacks=[csv_logger, checkpoint_logger])
	model.save('models/%s - final.h5' % version_name)

	# evaluate
	test_loss, test_acc = model.evaluate_generator(test_gen, steps=223)
	print('\n\033[1;34mtest_loss = %f\ntest_acc = %f' % (test_loss, test_acc), '\033[0m\n')
	evaluate_model(model, steps=30)
