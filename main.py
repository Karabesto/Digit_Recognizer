from data_preparation import DataManipulation

data_manip = DataManipulation()
# We read the dataset
dataset_train, dataset_test = data_manip.read_data()
# We prepare the data used for the training
images = data_manip.prep_data(dataset_train)
# We plot an image number_to_plot
number_to_plot = 10
# data_manip.display_image(images[number_to_plot])
data_manip.numeric_2_vectors_labels(dataset_train)
