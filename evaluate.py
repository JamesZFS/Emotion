import os

from keras import Sequential
from keras.models import load_model
from numpy import set_printoptions

from preprocess import load_dataset_from_file, generator_from_file_debug
from encoder import tag_list

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac


def evaluate_model(model: Sequential, eval_data_path: str = 'data/sina/sinanews.test', steps: int = 10):
	set_printoptions(precision=3, suppress=True)
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


if __name__ == '__main__':
	from train import version_name
	print(version_name)
	model = load_model('models/%s - best.h5' % version_name)
	assert isinstance(model, Sequential)
	model.summary()

	test_set = load_dataset_from_file('data/sina/sinanews.test')
	test_loss, test_acc = model.evaluate(test_set[0], test_set[1], batch_size=20)
	print('\n\033[1;34mtest_loss = %f\ntest_acc = %f' % (test_loss, test_acc), '\033[0m\n')

	evaluate_model(model, eval_data_path='data/sina/sinanews.demo', steps=8)

	evaluate_model(model, eval_data_path='data/sina/sinanews.train', steps=20)

	evaluate_model(model, eval_data_path='data/sina/sinanews.test', steps=30)
