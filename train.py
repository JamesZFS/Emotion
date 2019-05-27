import os

from keras import Sequential, layers, optimizers, callbacks
from keras.models import load_model

from evaluate import evaluate_model
from preprocess import load_dataset_from_file, \
	vocab_size, vec_dim, embedding_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac
# version_name = 'DeepBiLSTM1.0'
version_name = 'BiLSTM2.4'
# version_name = 'LSTM2.0'  # todo
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

	# opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01)
	opt = optimizers.Adam(lr=1e-4, epsilon=1e-8)
	# model.compile(opt, loss='mse', metrics=['acc'])
	model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
	model.summary()

	return model


def build_Bi_LSTM():
	'''build and compile an RNN models

	:return: RNN model
	'''
	dims = (300, 300, 8)
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=dims[0], mask_zero=True, input_length=1000,
							   weights=[embedding_matrix], trainable=False))
	model.add(layers.Bidirectional(layers.LSTM(units=dims[1], return_sequences=False)))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(units=dims[2], activation='softmax'))

	# opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.01)
	opt = optimizers.Adam(lr=1e-4, epsilon=1e-8)
	# model.compile(opt, loss='mse', metrics=['acc'])
	model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
	model.summary()

	return model


def build_Deep_Bi_LSTM():
	'''build and compile an RNN model # todo powerful now

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
		# layers.Bidirectional(layers.LSTM(vec_dim, return_sequences=True)),
		# layers.Dropout(0.4),
		layers.LSTM(200, return_sequences=False),
		layers.Dropout(0.4),
		layers.Dense(8, activation='softmax'),
	])

	opt = optimizers.RMSprop(lr=1e-4, rho=0.9, epsilon=None, decay=0.01)
	# model.compile(opt, loss='mse', metrics=['acc'])
	model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
	model.summary()

	return model


def build_CNN():
	'''build and compile a CNN model

	:return: CNN model
	'''
	model = Sequential()
	model.add(layers.Embedding(input_dim=vocab_size, output_dim=300, input_length=1000,
							   weights=[embedding_matrix], trainable=False))
	model.add(layers.Conv1D(filters=20, kernel_size=10, activation='relu'))
	model.add(layers.MaxPool1D(pool_size=2, strides=2))
	model.add(layers.Dropout(0.5))
	model.add(layers.Conv1D(filters=20, kernel_size=10, activation='relu'))
	model.add(layers.MaxPool1D(pool_size=10, strides=5))
	model.add(layers.Dropout(0.5))
	model.add(layers.Flatten())
	model.add(layers.Dense(units=8, activation='softmax'))

	opt = optimizers.Adam(lr=0.001, epsilon=1e-8)
	model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
	model.summary()

	return model


def build_MLP():
	'''build and compile an MLP model as baseline

	:return: MLP model
	'''
	model = Sequential([
		layers.Embedding(input_dim=vocab_size, output_dim=300, input_length=500,
						 weights=[embedding_matrix], trainable=False),
		layers.Flatten(),
		layers.Dense(500, activation='relu'),
		layers.Dropout(0.5),
		layers.Dense(100, activation='relu'),
		layers.Dropout(0.5),
		layers.Dense(8, activation='softmax'),
	])

	opt = optimizers.Adam(lr=0.0001, epsilon=1e-8)
	model.compile(opt, loss='categorical_crossentropy', metrics=['acc'])
	model.summary()

	return model


if __name__ == '__main__':
	# prepare data
	# split_train_val_from_file('data/sina/sinanews.train', output_dir='data/runtime', val_size=0.15)
	'''
	train_gen = generator_from_file('data/runtime/train', batch_size=10)  # around 1700 articles
	val_gen = generator_from_file('data/runtime/val', batch_size=10)  # around 350 articles
	test_gen = generator_from_file('data/runtime/test', batch_size=10)  # around 2000 articles
	'''
	train_set = load_dataset_from_file('data/sina/sinanews.train')
	# print(train_set[1][:5])

	# build model
	model = build_Bi_LSTM()
	# model = load_model('models/CNN1.2 - final.h5', compile=True)
	# assert isinstance(model, Sequential)

	csv_logger = callbacks.CSVLogger(
		'logs/%s.csv' % version_name,
		separator=',', append=False)
	checkpoint_logger = callbacks.ModelCheckpoint(
		'models/%s - best.h5' % version_name,
		monitor='val_loss', verbose=1, save_best_only=True)

	# train
	'''
	model.fit_generator(train_gen, epochs=5, steps_per_epoch=100,  # 100 x 10 = 1000 articles / epoch
						validation_data=val_gen, validation_steps=10,  # 10 x 10 = 100
						verbose=1, callbacks=[csv_logger, checkpoint_logger])
	'''
	model.fit(train_set[0], train_set[1], validation_split=0.10, epochs=5, batch_size=20,
			  verbose=1, callbacks=[csv_logger, checkpoint_logger])
	model.save('models/%s - final.h5' % version_name)

	# evaluate
	del train_set
	test_set = load_dataset_from_file('data/sina/sinanews.test')
	# test_loss, test_acc = model.evaluate_generator(test_gen, steps=223)
	test_loss, test_acc = model.evaluate(test_set[0], test_set[1], batch_size=20)
	print('\n\033[1;34mtest_loss = %f\ntest_acc = %f' % (test_loss, test_acc), '\033[0m\n')
	evaluate_model(model, steps=30)
