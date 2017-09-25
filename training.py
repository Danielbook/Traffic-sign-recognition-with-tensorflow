import os
import random
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# https://medium.com/@waleedka/traffic-sign-recognition-with-tensorflow-629dffc391a6

dir_path = os.path.dirname(os.path.realpath(__file__))


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


# Load training and testing datasets.
# train_data_dir = os.path.join(dir_path, "datasets/Germany/Training")
# test_data_dir = os.path.join(dir_path, "datasets/Germany/Testing")

train_data_dir = os.path.join(dir_path, 'datasets/BelgiumTS/Training')
test_data_dir = os.path.join(dir_path, "datasets/BelgiumTS/Testing")

images, labels = load_data(train_data_dir)

print("Unique Labels: {0}\nTotal Images: {1}".format(len(set(labels)), len(images)))

print("Images read")

# Resize images
images32 = [skimage.transform.resize(image, (32, 32))
            for image in images]

# display_images_and_labels(images32, labels)
print("Images resized")

labels_a = np.array(labels)
images_a = np.array(images32)

# Create a graph to hold the model.
graph = tf.Graph()

# Create model in the graph.
with graph.as_default():
    # Placeholders for inputs and labels.
    images_ph = tf.placeholder(tf.float32, [None, 32, 32, 3])
    labels_ph = tf.placeholder(tf.int32, [None])

    # Flatten input from: [None, height, width, channels]
    # To: [None, height * width * channels] == [None, 3072]
    images_flat = tf.contrib.layers.flatten(images_ph)

    # Fully connected layer.
    # Generates logits of size [None, 62]
    logits = tf.contrib.layers.fully_connected(images_flat, 62, tf.nn.relu)

    # Convert logits to label indexes (int).
    # Shape [None], which is a 1D vector of length == batch_size.
    predicted_labels = tf.argmax(logits, 1)

    # Define the loss function.
    # Cross-entropy is a good choice for classification.
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_ph))

    # Create training op.
    train = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    # And, finally, an initialization op to execute before training.
    init = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

print("Graph created")

# Create a session to run the graph we created.
session = tf.Session(graph=graph)

# First step is always to initialize all variables.
# We don't care about the return value, though. It's None.
_ = session.run(init)

for i in range(201):
    _, loss_value = session.run([train, loss],
                                feed_dict={images_ph: images_a, labels_ph: labels_a})
    if i % 10 == 0:
        print("Loss: ", loss_value)

# Save the variables to disk.
save_path = saver.save(session, dir_path + "/temp/model.ckpt")
print("Model saved in file: %s" % save_path)

session.close()
