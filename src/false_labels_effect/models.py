from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50


# define model
def create_cnn_model(img_shape, n_classes, false_labels_ratio):
    """
    Create a basic CNN

    Parameters
    ----------
    img_shape : tuple of two ints
        shape of image
    n_classes : int
        number of classes
    false_labels_ratio : float
        ratio of labels that are falsified

    Returns
    -------
    model : keras.Sequential
        cnn resnet model
    """
    activation_func = None
    kernel_tupel = (3, 3)
    pool_size = (2, 2)
    leaky_alpha = 0.3
    dropout_ratio_conv = 0.1
    dropout_ratio_dense = 0.1

    model = models.Sequential()

    model.add(layers.Conv2D(32,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.Conv2D(32,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Dropout(dropout_ratio_conv))
    model.add(layers.Conv2D(64,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.Conv2D(64,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Dropout(dropout_ratio_conv))
    model.add(layers.Conv2D(128,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.Conv2D(128,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Dropout(dropout_ratio_conv))
    model.add(layers.Conv2D(256,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Dropout(dropout_ratio_conv))
    model.add(layers.Conv2D(512,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Dropout(dropout_ratio_conv))
    model.add(layers.Conv2D(1024,
                            kernel_tupel,
                            activation=activation_func,
                            input_shape=(img_shape[0], img_shape[1], 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))
    model.add(layers.MaxPooling2D(pool_size))

    model.add(layers.Flatten())

    model.add(layers.Dropout(dropout_ratio_dense))
    model.add(layers.Dense(512))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))

    model.add(layers.Dropout(dropout_ratio_dense))
    model.add(layers.Dense(256))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=leaky_alpha))

    model.add(layers.Dropout(dropout_ratio_dense))
    model.add(layers.Dense(n_classes, activation='softmax'))

    model._name=f'basic_{format(int(false_labels_ratio*10000),"05d")}r_{n_classes}c'

    return model

def create_resnet_model(img_shape, n_classes, false_labels_ratio):
    """
    Create a CNN based on ResNet50

    Parameters
    ----------
    img_shape : tuple of two ints
        shape of image
    n_classes : int
        number of classes
    false_labels_ratio : float
        ratio of labels that are falsified

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

    model._name=f'resnet_{format(int(false_labels_ratio*10000),"05d")}r_{n_classes}c'

    return model
