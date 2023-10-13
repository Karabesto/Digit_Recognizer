import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DataManipulation:

    def __init__(self):
        self.data_train = None
        self.data_test = None
        self.images_train = None
        self.images_test = None
        self.labels_vector = None

    def read_data(self):
        """
        Load the data from the dataset.
        :return: Return an array with the first element being the dataset for training
                    and the second element the test's one
        """
        data_train = pd.read_csv('dataset/file_train.csv')
        data_test = pd.read_csv('dataset/file_test.csv')
        self.data_train = data_train
        self.data_test = data_test
        return data_train, data_test

    def prep_data(self):
        """
        Convert the data loaded in data used for learning (aka images on grayscale).
        :return: Return an array with the first element being the image for training
                    and the second element the test's one
        """
        # Each line is an image
        images = self.data_train.iloc[:, 1:].values
        # We force the type from int to float for each image
        images = images.astype(np.float64)
        # The data are encoded from 0 to 255 and we convert them to 0.0-1.0
        images = np.multiply(images, 1.0 / 255.0)
        self.images_test = images

        # Each line is an image
        images = self.data_test.iloc[:, 1:].values
        # We force the type from int to float for each image
        images = images.astype(np.float64)
        # The data are encoded from 0 to 255 and we convert them to 0.0-1.0
        images = np.multiply(images, 1.0 / 255.0)
        self.images_test = images
        return self.images_train, self.images_test

    @staticmethod
    def display_image(image):
        """
        Display an image given in parameter.
        :param image: image to plot
        """
        # The image size is the length of the row
        image_size = image.shape[0]
        # We define the height and width of the image with a square root because the images of the dataset are square
        image_width = np.ceil(np.sqrt(image_size)).astype(np.uint8)
        image_height = image_width
        print(f"Image size: {image_size}, image width: {image_width}, and image height: {image_height}")
        # We resize from 1d to 2d
        img = image.reshape(image_width, image_height)

        plt.axis('off')
        plt.imshow(img, cmap=cm.binary)
        plt.show()

    def digit_2_vectors_labels(self, dataset):
        """
        Convert a digit to a vectors used later,
        example : 0 -> [1 0 0 0 0 0 0 0 0 0],
                  1 -> [0 1 0 0 0 0 0 0 0 0],...
                  8 -> [0 0 0 0 0 0 0 0 1 0],...
        :param dataset:
        :return:
        """
        labels = dataset["label"]
        labels_vector = np.zeros((len(labels), 10))  # length of labels and 10 digits (0..9)
        for row_index, value in enumerate(labels):
            labels_vector[row_index, value] = 1
        self.labels_vector = labels_vector
        return labels_vector

    @staticmethod
    def vectors_2_digit_labels(vectors_labels):
        # Find the maximum value in each row
        if vectors_labels.ndim == 1:
            digit = np.argmax(vectors_labels)
        else:
            digit = np.argmax(vectors_labels, axis=1)
        return digit
