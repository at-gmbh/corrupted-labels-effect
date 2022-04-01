import tensorflow as tf

def set_visible_gpus(gpu_index: list):
    """
    Set visible GPUs to used for training

    Parameters
    ----------
    gpu_index : list of ints
        GPU devices to use for training by index

    Returns
    -------
    physical_gpus_use : list of objects
        GPU objects to use for training
    """
    physical_gpus = tf.config.list_physical_devices('GPU')
    physical_gpus_use = [physical_gpus[i] for i in gpu_index]

    if physical_gpus:
        # Restrict TensorFlow to only use the GPU as defined by passed index
        try:
            # prevent OOM due to GPU memory mapping
            for gpu in physical_gpus_use:
                tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.set_visible_devices(physical_gpus_use, 'GPU')

            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f'{len(physical_gpus)} Physical GPUs, {len(logical_gpus)} Logical GPU')

        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

        return physical_gpus_use
