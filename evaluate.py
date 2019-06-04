import os

import numpy as np
from keras import Sequential
from keras.models import load_model
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn import metrics

from config import *
from encoder import tag_list
from preprocess import load_dataset_from_file, generator_from_file_debug

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


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
	plt.hist(Y_true, bins=8)
	plt.title('Y_true dist')
	plt.show()
	plt.hist(Y_pred, bins=8)
	plt.title('Y_pred dist')
	plt.show()
	print(Y_true.shape)
	print(Y_pred.shape)
	acc = metrics.accuracy_score(Y_true, Y_pred)
	f1_micro = metrics.f1_score(Y_true, Y_pred, average='micro')
	f1_macro = metrics.f1_score(Y_true, Y_pred, average='macro')
	print('cov      =', cov)
	print('acc      =', acc)
	print('micro f1 =', f1_micro)
	print('macro f1 =', f1_macro)
	return cov, acc, f1_micro, f1_macro


if __name__ == '__main__':
	print('now testing', version_name)
	model = load_model('models/%s - best.h5' % version_name)
	assert isinstance(model, Sequential)
	model.summary()

	visualize_model(model, eval_data_path=test_file, steps=10)

	test_set = load_dataset_from_file(test_file)
	res = evaluate_model(model, test_set)
	with open('results/%s.txt' % version_name, 'w') as f:
		f.writelines([str(res[0]), '\n', str(res[1]), '\n', str(res[2]), '\n', str(res[3]), '\n'])
	print(version_name)
