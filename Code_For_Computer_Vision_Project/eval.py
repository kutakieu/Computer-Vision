# from __future__ import print_function
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from scipy import ndimage
from scipy.misc import imresize
import argparse
import shutil
from subprocess import call, PIPE
import sys
# from os.path import basename
# from os import makedir




FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('pixel_depth', 255.0, 'pixel depth')
tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint/', 'checkpoint dir')
tf.app.flags.DEFINE_integer('image_size', 48, 'input image width and height')

def to_letters_digits(i):
	return "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[i]

def main(filename):
	basename =  os.path.basename(filename)
	output_dir = "output/%s" % basename
	if os.path.isdir(output_dir):
		shutil.rmtree(output_dir)
	os.mkdir("output/"+basename)
	process = call(["./main", filename], stdout=PIPE, stderr=PIPE)
	# process
	# folder = "output/r6"
	# folder = "/home/dongtaoy/Desktop/test/data/Sample001"
	dir_list = sorted(os.listdir(output_dir))
	# print dir_list
	x_input = np.ndarray(
		shape=(len(dir_list), FLAGS.image_size, FLAGS.image_size,), 
		dtype=np.float32)
	for i, image_name in enumerate(dir_list):
		image_file = os.path.join(output_dir, image_name)

		image_data = imresize(ndimage.imread(image_file).astype(float), [48, 48])
		image_data = (image_data - FLAGS.pixel_depth / 2) / FLAGS.pixel_depth 
		# plt.imshow(image_data)
		# plt.show()
		x_input[i,:,:] = image_data
	x_input = x_input.reshape([-1, FLAGS.image_size, FLAGS.image_size, 1])
	checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
	graph = tf.Graph()
	with graph.as_default():
		# session_conf = tf.ConfigProto()
		sess = tf.Session()
		with sess.as_default():
			saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
	        saver.restore(sess, checkpoint_file)
	        x = graph.get_operation_by_name("input/x-input").outputs[0]
	        dropout_keep_prob = graph.get_operation_by_name("input/dropout_keep_probability").outputs[0]
	        y = graph.get_operation_by_name("output/Wx_plus_b/predictions").outputs[0]

	      	predictions = sess.run(y, feed_dict={x:x_input, dropout_keep_prob:1.0})
	      	for row in predictions:
	      		sys.stdout.write(to_letters_digits(np.argmax(row)))
	      		# print(, end="")
	      	print

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--file", dest="filename",
		help="file to be processed", metavar="FILE")
	args = parser.parse_args()
	if not args.filename:
		parser.print_help()
		exit()

	if os.path.isfile(args.filename):
		main(args.filename)
	else:
		print "file %s does not exist" % args.filename
	