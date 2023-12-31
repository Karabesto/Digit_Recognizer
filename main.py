from data_preparation import DataManipulation

data_manip = DataManipulation()
# We read the dataset
dataset_train, dataset_test = data_manip.read_data()
# We prepare the data used for the training
images_train, image_test = data_manip.prep_data()
# We plot an image number_to_plot
number_to_plot = 10
# data_manip.display_image(images_train[number_to_plot])
labels_vect = data_manip.digit_2_vectors_labels(dataset_train)
labels = data_manip.vectors_2_digit_labels(labels_vect)
labels = data_manip.vectors_2_digit_labels(labels_vect[0])
