import os

import keras.backend.tensorflow_backend as KTF
import tensorflow as tf
from keras import Sequential, layers, optimizers, callbacks

from config import *
from evaluate import evaluate_model
from preprocess import load_dataset_from_file, \
	vocab_size, vec_dim, embedding_matrix

if must_use_cpu:
	num_cores = 4
	num_CPU = 1
	num_GPU = 0
	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
							inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
							device_count={'CPU': num_CPU, 'GPU': num_GPU})
	session = tf.Session(config=config)
	KTF.set_session(session)

if method == 'norm':
	loss = 'mse'
elif method == 'one hot':
	loss = 'categorical_crossentropy'
else:
	raise ValueError('method should be either \'norm\' or \'one hot\'')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

assert vec_dim == 300


def build_LSTM():
	'''build and compile an RNN models

	:return: RNN model
	'''
	dims = (300, 200, 8)
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=dims[0], mask_zero=True, input_length=1000,
							   weights=[embedding_matrix], trainable=False))
	model.add(layers.LSTM(units=dims[1], return_sequences=False))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(units=dims[2], activation='softmax'))

	opt = optimizers.Adam(lr=1e-4, epsilon=1e-8)
	model.compile(opt, loss=loss, metrics=['acc'])
	model.summary()

	return model


def build_Bi_LSTM():
	'''build and compile an RNN model

	:return: RNN model
	'''
	dims = (300, 300, 8)
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=dims[0], mask_zero=True, input_length=1000,
							   weights=[embedding_matrix], trainable=False))
	model.add(layers.Bidirectional(layers.LSTM(units=dims[1], return_sequences=False)))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(units=dims[2], activation='softmax'))

	opt = optimizers.Adam(lr=1e-4, epsilon=1e-8)
	model.compile(opt, loss=loss, metrics=['acc'])
	model.summary()

	return model


def build_Deep_Bi_LSTM():
	'''build and compile an RNN model

	:return: RNN model
	'''
	sequence_len = 1000
	assert vec_dim == 300
	model = Sequential([
		layers.Embedding(input_dim=vocab_size, output_dim=vec_dim, input_length=sequence_len, mask_zero=True,
						 weights=[embedding_matrix], trainable=False),
		layers.Bidirectional(layers.LSTM(vec_dim, input_shape=(sequence_len, vec_dim), return_sequences=True)),
		layers.Dropout(0.4),
		layers.Bidirectional(layers.LSTM(vec_dim, return_sequences=True)),
		layers.Dropout(0.4),
		layers.LSTM(200, return_sequences=False),
		layers.Dropout(0.4),
		layers.Dense(8, activation='softmax'),
	])

	opt = optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.01)
	model.compile(opt, loss=loss, metrics=['acc'])
	model.summary()

	return model


def build_CNN():
	'''build and compile a CNN model

	:return: CNN model
	'''
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=300, input_length=1000,
							   weights=[embedding_matrix], trainable=False))
	model.add(layers.Conv1D(filters=200, kernel_size=10, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Flatten())
	model.add(layers.Dense(units=8, activation='softmax'))

	opt = optimizers.Adam(lr=1e-4, epsilon=1e-8)
	model.compile(opt, loss=loss, metrics=['acc'])
	model.summary()

	return model


def build_MLP():
	'''build and compile an MLP model as baseline

	:return: MLP model
	'''
	model = Sequential([
		layers.Embedding(input_dim=vocab_size, output_dim=300, input_length=1000,
						 weights=[embedding_matrix], trainable=False),
		layers.Flatten(),
		layers.Dense(200, activation='relu'),
		layers.Dropout(0.5),
		layers.Dense(8, activation='softmax'),
	])

	opt = optimizers.Adam(lr=0.0001, epsilon=1e-8)
	model.compile(opt, loss=loss, metrics=['acc'])
	model.summary()

	return model


if __name__ == '__main__':
	print('now training', version_name)
	# prepare data
	train_set = load_dataset_from_file(train_file, shuffle=False)

	# build model
	if build_type == 'MLP':
		model = build_MLP()
	elif build_type == 'RNN':
		model = build_Bi_LSTM()
	elif build_type == 'CNN':
		model = build_CNN()
	else:
		raise ValueError('build_type should be either \'MLP\' or \'RNN\' or \'CNN\'')

	# model = load_model('models/%s - final.h5' % version_name, compile=True)
	# assert isinstance(model, Sequential)

	csv_logger = callbacks.CSVLogger(
		'logs/%s.csv' % version_name,
		separator=',', append=False)
	checkpoint_logger = callbacks.ModelCheckpoint(
		'models/%s - best.h5' % version_name,
		monitor='val_loss', verbose=1, save_best_only=True)
	early_stopping = callbacks.EarlyStopping(patience=2)

	# train
	model.fit(train_set[0], train_set[1], validation_split=0.10, epochs=5, batch_size=20, initial_epoch=0,
			  verbose=1, callbacks=[csv_logger, checkpoint_logger, early_stopping])
	model.save('models/%s - final.h5' % version_name)

	# evaluate best model
	del train_set
	test_set = load_dataset_from_file(test_file)
	model.load_weights('models/%s - best.h5' % version_name)
	res = evaluate_model(model, test_set)
	with open('results/%s.txt' % version_name, 'w') as f:
		f.writelines([str(res[0]), '\n', str(res[1]), '\n', str(res[2]), '\n', str(res[3]), '\n'])
	print(version_name)
