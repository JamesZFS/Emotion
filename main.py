import os
import re
import torch
import sys
# from keras import Sequential, layers, optimizers, utils, preprocessing

# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # todo this is important on mac

if __name__ == '__main__':
	print(sys.path)
	os.chdir('data/sina')
	with open('sinanews.demo', 'r') as f:
		lines = f.readlines()

	print(lines[0])
