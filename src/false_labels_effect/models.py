from tensorflow.keras import layers, models


# define model
def create_cnn_model(shape):
    """
    Create a basic CNN

    Parameters
    ----------
    shape : tuple of two ints
        shape of image

    Returns
    -------
    cnn_model : keras.Sequential
        CNN with three Conv2D and two Dense layers
    """
    cnn_model = models.Sequential()

    cnn_model.add(layers.Conv2D(244, (3, 3), activation='relu', input_shape=(shape[0], shape[1], 3)))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(488, (3, 3), activation='relu', input_shape=(shape[0], shape[1], 3)))
    cnn_model.add(layers.MaxPooling2D((2, 2)))
    cnn_model.add(layers.Conv2D(978, (3, 3), activation='relu', input_shape=(shape[0], shape[1], 3)))

    cnn_model.add(layers.Flatten())
    cnn_model.add(layers.Dense(64, activation='relu'))
    cnn_model.add(layers.Dense(4, activation='softmax'))

    # print(cnn_model.summary())

    return cnn_model
