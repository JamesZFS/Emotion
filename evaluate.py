import os

from keras import Sequential
from keras.models import load_model
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr

from preprocess import load_dataset_from_file, generator_from_file_debug
from encoder import tag_list

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac


def visualize_model(model: Sequential, eval_data_path: str = 'data/sina/sinanews.test', steps: int = 10):
	np.set_printoptions(precision=3, suppress=True)
	data_gen = generator_from_file_debug(eval_data_path, batch_size=1)
	print('\033[95mEvaluating on %s\nTags: %s\033[0m' % (os.path.basename(eval_data_path), ' '.join(tag_list)))
	correct = 0
	tot = 0
	for _ in range(steps):
		X, Y, article = next(data_gen)
		pred_proba = model.predict(X, 1)
		truth = tag_list[Y.argmax()]
		pred = tag_list[pred_proba.argmax()]
		print(article[0].strip())
		print('truth: ', truth, Y)
		tot += 1
		if truth == pred:
			correct += 1
			print('\033[32m', end='')
		else:
			print('\033[31m', end='')
		print('pred:  ', pred, pred_proba, end='')
		print('\033[0m\n')

	acc = correct / tot
	print('\033[1;34maccuracy:', acc, '\033[0m\n\n')
	return acc


def evaluate_model(model: Sequential, eval_data: tuple):
	X, Y_true = eval_data
	assert isinstance(X, np.ndarray)
	assert isinstance(Y_true, np.ndarray)
	Y_pred = model.predict(X, batch_size=10, verbose=1)
	print(Y_true.shape)
	print(Y_pred.shape)
	cov = np.mean([pearsonr(x, y)[0] for x, y in zip(Y_true, Y_pred)])

	Y_true = np.argmax(Y_true, axis=1)
	Y_pred = np.argmax(Y_pred, axis=1)
	print(Y_true.shape)
	print(Y_pred.shape)
	acc = metrics.accuracy_score(Y_true, Y_pred)
	f1_micro = metrics.f1_score(Y_true, Y_pred, average='micro')
	f1_macro = metrics.f1_score(Y_true, Y_pred, average='macro')
	print('cov      =', cov)
	print('acc      =', acc)
	print('micro f1 =', f1_micro)
	print('macro f1 =', f1_macro)
	return acc, f1_micro, f1_macro


if __name__ == '__main__':
	# from train import version_name
	version_name = 'CNN2.1'
	print(version_name)
	model = load_model('models/%s - best.h5' % version_name)
	assert isinstance(model, Sequential)
	model.summary()

	test_set = load_dataset_from_file('data/sina/sinanews.test')
	evaluate_model(model, test_set)
	# test_loss, test_acc = model.evaluate(test_set[0], test_set[1], batch_size=20)
	# print('\n\033[1;34mtest_loss = %f\ntest_acc = %f' % (test_loss, test_acc), '\033[0m\n')

	# visualize_model(model, eval_data_path='data/sina/sinanews.demo', steps=8)

	# visualize_model(model, eval_data_path='data/sina/sinanews.train', steps=20)

	# visualize_model(model, eval_data_path='data/sina/sinanews.test', steps=30)
