import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm


class DataManipulation:
    @staticmethod
    def read_data():
        data_train = pd.read_csv('dataset/file_train.csv')
        data_test = pd.read_csv('dataset/file_test.csv')
        return data_train, data_test

    @staticmethod
    def prep_data(data):
        # Each line is an image
        images = data.iloc[:, 1:].values
        print(images[0])
        # We force the type from int to float for each image
        images = images.astype(np.float64)
        # The data are encoded from 0 to 255 and we convert them to 0.0-1.0
        images = np.multiply(images, 1.0 / 255.0)
        return images

    @staticmethod
    def display_image(image):
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

    @staticmethod
    def numeric_2_vectors_labels(dataset):
        print(dataset.loc[0])

