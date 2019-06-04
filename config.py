# version_name = 'RNN_norm'		# best version
# version_name = 'RNN_one_hot'
version_name = 'CNN_norm'
# version_name = 'CNN_one_hot'
# version_name = 'MLP_norm'
# version_name = 'MLP_one_hot'

# method = 'one hot' # todo this should be corresponding to version_name
method = 'norm'

train_file = 'data/sina/sinanews.train'
test_file = 'data/sina/sinanews.test'

build_type = 'MLP'
# build_type = 'CNN'
# build_type = 'RNN'

must_use_cpu = False # don't change this by default
