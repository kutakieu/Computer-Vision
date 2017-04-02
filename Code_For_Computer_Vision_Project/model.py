import tensorflow as tf
import numpy as np
import os
from six.moves import cPickle as pickle
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.misc import imresize
import time
from datetime import datetime

DEBUG = True
DATA_ROOT_DIR = 'data/'
PICKLE_FILE = 'data_small.pickle'

# flags = tf.app.flags
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('batch_size', 50, 'input batch size')
tf.app.flags.DEFINE_integer('image_size', 48, 'input image width and height')
tf.app.flags.DEFINE_integer('image_channel', 1, 'input image channel')
tf.app.flags.DEFINE_integer('num_classes', 36, 'number of output classes')
tf.app.flags.DEFINE_integer('max_steps', 25000+1, 'number of steps to run trainer.')
# flags.DEFINE_integer('pixel_depth', 255, 'pixel depth')
tf.app.flags.DEFINE_integer('max_num_images', 40000, 'max number of input images')
tf.app.flags.DEFINE_integer('min_num_images', 1000, 'min number of input images')
tf.app.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps")
tf.app.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps ")
tf.app.flags.DEFINE_integer("log_every", 20, "log_every ")

tf.app.flags.DEFINE_integer('training_dataset_ratio', 0.6, 'training_dataset_ratio')
tf.app.flags.DEFINE_integer('validation_dataset_ratio', 0.2, 'validation_dataset_ratio')


tf.app.flags.DEFINE_integer('cnn_1_depth', 32, 'cnn_1_depth')
tf.app.flags.DEFINE_integer('cnn_1_path_size', 5, 'cnn_1_path_size')
tf.app.flags.DEFINE_integer('cnn_2_depth', 64, 'cnn_2_depth')
tf.app.flags.DEFINE_integer('cnn_2_patch_size', 5, 'cnn_2_depth')
tf.app.flags.DEFINE_integer('nn_1_depth', 128, 'nn_1_depth')

tf.app.flags.DEFINE_float('pixel_depth', 255.0, 'pixel depth')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')
# flags.DEFINE_float('dropout', 0.9, 'Keep probability for training dropout.')
tf.app.flags.DEFINE_string('data_root_dir', 'data/', 'Directory for storing data')
tf.app.flags.DEFINE_string('summaries_dir', 'summary/', 'Summaries directory')
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', 'checkpoint dir')



np.random.seed(1234)

def init_load(data_folders):
	dataset = np.ndarray(
		shape=(FLAGS.max_num_images, FLAGS.image_size, FLAGS.image_size,), 
		dtype=np.float32)
	labels = np.ndarray(
		shape=(FLAGS.max_num_images,), 
		dtype=np.int32)
	label_index = 0
	image_index = 0
	for folder in sorted(data_folders):
		# count = 0
		if (DEBUG):
			print folder
		for image_name in sorted(os.listdir(folder)):
			# if count
			image_file = os.path.join(folder, image_name)
			try:
				image_data = imresize((ndimage.imread(image_file).astype(float)), [FLAGS.image_size, FLAGS.image_size])
				image_data = (image_data - FLAGS.pixel_depth / 2) / FLAGS.pixel_depth
				if image_data.shape != (FLAGS.image_size, FLAGS.image_size):
					raise Exception('Unexpected image shape: %s' % str(image_data.shape))
				dataset[image_index, :, :] = image_data
				labels[image_index] = label_index
				image_index += 1
				# plt.imshow(image_data)
				# plt.show()
			except IOError as e:
				print('Could not read:', image_file, ':', e, '- it\'s ok, skipping.')
		label_index += 1
		# break
		# break
	num_images = image_index
	dataset = dataset[0:num_images, :, :]
	labels = labels[0:num_images]
	if num_images < FLAGS.min_num_images:
		raise Exception('Many fewer images than expected: %d < %d' % (num_images, FLAGS.min_num_images))
	if (DEBUG):
		print('Full dataset tensor:', dataset.shape)
		print('Mean:', np.mean(dataset))
		print('Labels:', labels.shape)
	return dataset, labels


def to_letters_digits(i):
	return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]


def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation,:,:]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels


def save(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels):
	try:
		print "Saving data..."
		f = open(PICKLE_FILE, 'wb')
		save = {
			'train_dataset': train_dataset,
			'train_labels': train_labels,
			'valid_dataset': valid_dataset,
			'valid_labels': valid_labels,
			'test_dataset': test_dataset,
			'test_labels': test_labels,
		}
		pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
		f.close()
		if (DEBUG):
			statinfo = os.stat(PICKLE_FILE)
			print('Compressed pickle size:', statinfo.st_size)
	except Exception as e:
		print('Unable to save data to', PICKLE_FILE, ':', e)
		raise

def to_letters_digits(i):
	return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]

def load():
	try:
		with open(PICKLE_FILE, 'rb') as f:
			print "Loading Data..."
			save = pickle.load(f)
			train_dataset = save['train_dataset']
			train_labels = save['train_labels']
			valid_dataset = save['valid_dataset']
			valid_labels = save['valid_labels']
			test_dataset = save['test_dataset']
			test_labels = save['test_labels']
			del save  # hint to help gc free up memory
			print('Training set', train_dataset.shape, train_labels.shape)
			print('Validation set', valid_dataset.shape, valid_labels.shape)
			print('Test set', test_dataset.shape, test_labels.shape)
		return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels
	except Exception as e:
		print('Unable to load data from', PICKLE_FILE, ':', e)
		raise


def construct_data():
	print "Constructing data..."
	data_folders = [os.path.join(FLAGS.data_root_dir,o) for o in os.listdir(FLAGS.data_root_dir) if os.path.isdir(os.path.join(FLAGS.data_root_dir,o))]
	dataset, labels = init_load(data_folders)
	# for i in range(36):
	# 	plt.imshow(dataset[i*1016].reshape(FLAGS.image_size, FLAGS.image_size))
	# 	plt.title(to_letters_digits(labels[i*1016]))
	# 	plt.show()
	dataset, labels = randomize(dataset, labels)
	num_data_instance = dataset.shape[0]
	
	num_training_instance = int(num_data_instance * FLAGS.training_dataset_ratio)
	num_validation_instance = int(num_data_instance * FLAGS.validation_dataset_ratio)
	num_testing_instance = num_data_instance - num_training_instance - num_validation_instance

	if (DEBUG):
		print "num_data_instance %d" % num_data_instance
		print "num_training_instance %d" % num_training_instance
		print "num_validation_instance %d" % num_validation_instance
		print "num_testing_instance %d" % num_testing_instance

	train_dataset = dataset[:num_training_instance,:,:]
	train_labels = labels[:num_training_instance]

	valid_dataset = dataset[num_training_instance:num_training_instance+num_validation_instance,:,:]
	valid_labels = labels[num_training_instance:num_training_instance+num_validation_instance]

	test_dataset = dataset[num_training_instance+num_validation_instance:num_data_instance,:,:]
	test_labels = labels[num_training_instance+num_validation_instance:num_data_instance]

	if (DEBUG):
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)

	save(train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels)
	return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels


# def reformat(dataset, labels):
# 	dataset = dataset.reshape((-1, IMAGE_SIZE * IMAGE_SIZE)).astype(np.float32)
# 	labels = (np.arange(NUM_CLASSES) == labels[:,None]).astype(np.float32)
# 	return dataset, labels

def reformat(dataset, labels):
	dataset = dataset.reshape((-1, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel)).astype(np.float32)
	labels = (np.arange(FLAGS.num_classes) == labels[:,None]).astype(np.float32)
	return dataset, labels









def main():

	try:
		train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = load()
	except Exception as e:
		print e
		train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = construct_data()

	train_dataset, train_labels = reformat(train_dataset, train_labels)
	valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
	test_dataset, test_labels = reformat(test_dataset, test_labels)
	if (DEBUG):
		print('Training set', train_dataset.shape, train_labels.shape)
		print('Validation set', valid_dataset.shape, valid_labels.shape)
		print('Test set', test_dataset.shape, test_labels.shape)

	if not os.path.exists(FLAGS.checkpoint_dir):
		os.makedirs(FLAGS.checkpoint_dir)
	
	graph = tf.Graph()

	with graph.as_default():
		def weight_variable(shape):
			"""Create a weight variable with appropriate initialization."""
			initial = tf.truncated_normal(shape, stddev=0.1)
			return tf.Variable(initial)


		def bias_variable(shape):
			"""Create a bias variable with appropriate initialization."""
			initial = tf.constant(0.1, shape=shape)
			return tf.Variable(initial)


		def variable_summaries(var, name):
			"""Attach a lot of summaries to a Tensor."""
			with tf.name_scope('summaries'):
				mean = tf.reduce_mean(var)
				tf.scalar_summary('mean/' + name, mean)
				with tf.name_scope('stddev'):
					stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
				tf.scalar_summary('sttdev/' + name, stddev)
				tf.scalar_summary('max/' + name, tf.reduce_max(var))
				tf.scalar_summary('min/' + name, tf.reduce_min(var))
				tf.histogram_summary(name, var)


		def cnn_layer(input_tensor, patch_size, input_dim, output_dim, layer_name, act=tf.nn.relu):
			"""Reusable code for making a simple neural net layer.
			It does a matrix multiply, bias add, and then uses relu to nonlinearize.
			It also sets up name scoping so that the resultant graph is easy to read, and
			adds a number of summary ops.
			"""
			# Adding a name scope ensures logical grouping of the layers in the graph.
			with tf.name_scope(layer_name):
				# This Variable will hold the state of the weights for the layer
				with tf.name_scope('weights'):
					weights = weight_variable([patch_size, patch_size, input_dim, output_dim])
					variable_summaries(weights, layer_name + '/weights')
				with tf.name_scope('biases'):
					biases = bias_variable([output_dim])
					variable_summaries(biases, layer_name + '/biases')
				with tf.name_scope('Wx_plus_b'):
					conv = tf.nn.conv2d(input_tensor, weights, [1, 1, 1, 1], padding='SAME')
					wx_plus_b = conv + biases
					tf.histogram_summary(layer_name + '/Wx_plus_b', wx_plus_b)
				with tf.name_scope('max_pool'):
					max_pool = tf.nn.max_pool(wx_plus_b, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
					tf.histogram_summary(layer_name + '/max_pool', max_pool)

				activations = act(max_pool, 'activation')
				tf.histogram_summary(layer_name + '/activations', activations)
				return activations


		def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
			"""Reusable code for making a simple neural net layer.
			It does a matrix multiply, bias add, and then uses relu to nonlinearize.
			It also sets up name scoping so that the resultant graph is easy to read, and
			adds a number of summary ops.
			"""
			# Adding a name scope ensures logical grouping of the layers in the graph.
			with tf.name_scope(layer_name):
				# This Variable will hold the state of the weights for the layer
				with tf.name_scope('weights'):
					weights = weight_variable([input_dim, output_dim])
					variable_summaries(weights, layer_name + '/weights')
				with tf.name_scope('biases'):
					biases = bias_variable([output_dim])
					variable_summaries(biases, layer_name + '/biases')
				with tf.name_scope('Wx_plus_b'):
					preactivate = tf.add(tf.matmul(input_tensor, weights), biases, name='predictions')
					tf.histogram_summary(layer_name + '/pre_activations', preactivate)
				if act == None:
					return preactivate
				else:
					return act(preactivate, 'activation')
				# tf.histogram_summary(layer_name + '/activations', activations)
				# return activations

		with tf.name_scope('input'):
			x = tf.placeholder(tf.float32, shape=(None, FLAGS.image_size, FLAGS.image_size, FLAGS.image_channel), name="x-input")
			tf.image_summary('input', x, 10)
			y_ = tf.placeholder(tf.float32, shape=(None, FLAGS.num_classes), name="y-input")
			keep_prob = tf.placeholder(tf.float32, name="dropout_keep_probability")
			tf.scalar_summary('dropout_keep_probability', keep_prob)

		# with tf.name_scope('global_step'):
		global_step = tf.Variable(0, name='global_step', trainable=False)
			# train_op = optimizer.minimize(loss, global_step=global_step)

		# def model(data):
		hidden1 = cnn_layer(x, FLAGS.cnn_1_path_size, FLAGS.image_channel, FLAGS.cnn_1_depth, "CNN1")
		hidden2 = cnn_layer(hidden1, FLAGS.cnn_2_patch_size, FLAGS.cnn_1_depth, FLAGS.cnn_2_depth, "CNN2")
		shape = hidden2.get_shape().as_list()
		# depth = shape[1] * shape[2] * shape[3]
		reshape = tf.reshape(hidden2, [-1, shape[1] * shape[2] * shape[3]])
		hidden3 = nn_layer(reshape, shape[1] * shape[2] * shape[3], 128, 'NN1')
		drop = tf.nn.dropout(hidden3, keep_prob)
		y = nn_layer(drop, FLAGS.nn_1_depth, FLAGS.num_classes, 'output', act=None)
			# return y

		# y = model(x)
		# y_ = labels = tf.cast(y_, tf.int64)
		with tf.name_scope('cross_entropy'):
			cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
			# diff = y_ * tf.log(y)
			# with tf.name_scope('total'):
				# cross_entropy = -tf.reduce_mean(diff)
			# tf.scalar_summary('cross entropy', cross_entropy)

		with tf.name_scope('train'):
			train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy, global_step=global_step)


		with tf.name_scope('accuracy'):
			with tf.name_scope('correct_prediction'):
				correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
			with tf.name_scope('accuracy'):
				accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) * 100
			tf.scalar_summary('accuracy', accuracy)

		# valid_prediction = tf.nn.softmax(model(tf_valid_dataset))

		merged = tf.merge_all_summaries()
		saver = tf.train.Saver()
		with tf.Session(graph=graph) as sess:
			ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				tf.initialize_all_variables().run()
			train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
			validate_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/validate', sess.graph)
			for step in range(FLAGS.max_steps):
				current_step = tf.train.global_step(sess, global_step)
				start_time = time.time()
				offset = (current_step * FLAGS.batch_size) % (train_labels.shape[0] - FLAGS.batch_size)
				batch_data = train_dataset[offset:(offset + FLAGS.batch_size), :, :, :]
				batch_labels = train_labels[offset:(offset + FLAGS.batch_size), :]
				feed_dict = {x : batch_data, y_ : batch_labels, keep_prob: 0.5}
				
				summary, _, y_value, accuracy_value, cross_entropy_value, _ = sess.run(
					[merged, train_op, y, accuracy, cross_entropy, global_step], feed_dict=feed_dict)
				train_writer.add_summary(summary, current_step)
				duration = time.time() - start_time
				if current_step % FLAGS.log_every == 0:
					num_examples_per_step = FLAGS.batch_size
					examples_per_sec = num_examples_per_step / duration
					sec_per_batch = float(duration)

					format_str = ('%s: training step %d, acc=%.2f%% loss = %.2f  (%.1f examples/sec; %.3f ''sec/batch)')
					print (format_str % (datetime.now(), current_step, accuracy_value, cross_entropy_value,
					                     examples_per_sec, sec_per_batch))

				
				if current_step % FLAGS.evaluate_every == 0:
					feed_dict = {x:valid_dataset, y_:valid_labels, keep_prob: 1}
					summary, accuracy_value = sess.run([merged, accuracy], feed_dict=feed_dict)
					# accuracy_value = accuracy.eval(feed_dict=feed_dict)
					print ('%s: validation step %d, acc=%.2f%%') % (datetime.now(), current_step, accuracy_value)
					validate_writer.add_summary(summary, current_step)


				if current_step % FLAGS.checkpoint_every == 0:
					path = saver.save(sess, FLAGS.checkpoint_dir + 'model.ckpt', global_step=current_step)
					print "Saved model checkpoint {} to {}".format(current_step, path)
			
				

if __name__ == '__main__':

	main()

# class AlphanumericCNN(object):


# 	def __init__():
# 		self.x = tf.placeholder(tf.float32, shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
# 		self.y = tf.placeholder(tf.float32, shape=(batch_size, NUM_CLASSES))




# 	def nn_layer(input_tensor, input_dim, output_dim, layer_name):
# 	"""Reusable code for making a simple neural net layer.

# 	It does a matrix multiply, bias add, and then uses relu to nonlinearize.
# 	It also sets up name scoping so that the resultant graph is easy to read, and
# 	adds a number of summary ops.
# 	"""
# 	# Adding a name scope ensures logical grouping of the layers in the graph.
# 	with tf.name_scope(layer_name):
# 		# This Variable will hold the state of the weights for the layer
# 		with tf.name_scope("weights"):
# 			weights = weight_variable([input_dim, output_dim])
# 			variable_summaries(weights, layer_name + '/weights')
# 		with tf.name_scope("biases"):
# 			biases = bias_variable([output_dim])
# 			variable_summaries(biases, layer_name + '/biases')
# 		with tf.name_scope('Wx_plus_b'):
# 			activations = tf.matmul(input_tensor, weights) + biases
# 			tf.histogram_summary(layer_name + '/activations', activations)
# 		relu = tf.nn.relu(activations, 'relu')
# 		tf.histogram_summary(layer_name + '/activations_relu', relu)
# 		return tf.nn.dropout(relu, keep_prob)