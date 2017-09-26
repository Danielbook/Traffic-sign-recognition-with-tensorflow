import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6
# https://github.com/waleedka/traffic-signs-tensorflow/blob/master/notebook1.ipynb

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

# Load testing datasets.
TEST_DATA_DIR = os.path.join(DIR_PATH, 'datasets/BelgiumTS/Testing')
MODEL_PATH = os.path.join(DIR_PATH, 'temp/model-1000.meta')
IMG_SIZE = 32


def load_data(data_dir):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    """
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f)
                      for f in os.listdir(label_dir) if f.endswith(".ppm")]
        # For each label, load it's images and add them to the images list.
        # And add the label number (i.e. directory name) to the labels list.
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels


# Load the test dataset.
test_images, test_labels = load_data(TEST_DATA_DIR)

print("Test images loaded")

# Transform the images, just like we did with the training set.
test_images32 = [skimage.transform.resize(image, (IMG_SIZE, IMG_SIZE))
                 for image in test_images]

print("Images resized")

# Add ops to save and restore all the variables.
new_saver = tf.train.import_meta_graph(MODEL_PATH)

# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
    # Restore variables from disk.
    new_saver.restore(sess, tf.train.latest_checkpoint(DIR_PATH+'/temp/'))
    # saver.restore(sess, MODEL_PATH)
    print("Model restored.")
    graph = tf.get_default_graph()
    predicted_labels = graph.get_tensor_by_name('predicted_labels:0')
    images_ph = graph.get_tensor_by_name('images_ph:0')

    # Run predictions against the full test set.
    predicted = sess.run([predicted_labels],
                         feed_dict={images_ph: test_images32})[0]

    # Calculate how many matches we got.
    match_count = sum([int(y == y_) for y, y_ in zip(test_labels, predicted)])
    accuracy = match_count / len(test_labels)
    print("Accuracy: {:.3f}".format(accuracy))
    sess.close()
