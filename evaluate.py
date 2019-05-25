import os
from keras import Sequential

from preprocess import tag_list, generator_from_file_debug

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac


def evaluate_model(model: Sequential, eval_data_path: str = 'data/sina/sinanews.demo', steps: int = 10):
	# data_gen = generator_from_file(eval_data_path, batch_size=1)
	# res = model.predict_generator(data_gen, steps)

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
	from train import build_RNN
	from numpy import set_printoptions

	model = build_RNN()
	model.load_weights('models/state1.0.h5')

	set_printoptions(precision=3, suppress=True)

	evaluate_model(model, eval_data_path='data/sina/sinanews.demo', steps=8)

	evaluate_model(model, eval_data_path='data/sina/sinanews.train', steps=20)

	evaluate_model(model, eval_data_path='data/sina/sinanews.test', steps=20)

