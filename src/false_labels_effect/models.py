from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


# define model
def create_cnn_model(img_shape, n_classes):
    """
    Create a basic CNN

    Parameters
    ----------
    img_shape : tuple of two ints
        shape of image
    n_classes : int
        number of classes

    Returns
    -------
    model : keras.Sequential
        cnn resnet model
    """
    model = models.Sequential()

    model.add(layers.Conv2D(244, (3, 3), activation='relu', input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(488, (3, 3), activation='relu', input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(978, (3, 3), activation='relu', input_shape=(img_shape[0], img_shape[1], 3)))

    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model._name="basic_cnn"

    print(model.summary())

    return model


def create_resnet_model(img_shape, n_classes):
    """
    Create a CNN based on ResNet50

    Parameters
    ----------
    img_shape : tuple
        shape of image
    n_classes : int
        number of classes

    Returns
    -------
    model : keras.Sequential
        cnn resnet model
    """
    resnet_model = ResNet50(include_top=False, weights="imagenet", input_shape=(img_shape[0], img_shape[1], 3))

    model = models.Sequential()

    model.add(resnet_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(2048, activation="relu"))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model._name="resnet50_cnn"

    print(model.summary())

    return model
