import json
import os

import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback


class class_report_cb(Callback):
    """
    Calculates and logs classification report for test data after each epoch

    Parameters
    ----------
    dataloader : DataLoader
        keras dataloader object

    Returns
    -------
    class_report : dict
        dictionary of classification report
    """
    def __init__(self, dataloader, model_start_time):
        self.dataloader = dataloader
        self.model_start_time = model_start_time
    
    def on_epoch_begin(self, epoch, logs=None):
        self.dataloader.y_true_dict = {}

    def on_epoch_end(self, epoch, logs=None):

        predict = self.model.predict(self.dataloader)
        y_pred = tf.argmax(predict, axis=1)
        
        y_true = list(self.dataloader.y_true_dict.values())

        class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)

        # initialize logging
        logdir_class_report = f'./logs/class_report/{self.model._name}/{self.model_start_time}'

        if not os.path.exists(logdir_class_report):
            os.makedirs(logdir_class_report)
        with open(logdir_class_report + f'/class_report_{epoch}epoch.json', 'w+') as f:
            json.dump(class_report, f)

        for i in range(self.dataloader.n_classes):
            tf.summary.scalar(f'epoch_precision_label_{i}', data=class_report[f'{i}']['precision'], step=epoch)
            tf.summary.scalar(f'epoch_recall_label_{i}', data=class_report[f'{i}']['recall'], step=epoch)
            tf.summary.scalar(f'epoch_f1-score_label_{i}', data=class_report[f'{i}']['f1-score'], step=epoch)

        tf.summary.scalar(f'epoch_accuracy', data=class_report['accuracy'], step=epoch)
        tf.summary.scalar(f'epoch_precision_macro', data=class_report['macro avg']['precision'], step=epoch)
        tf.summary.scalar(f'epoch_recall_macro', data=class_report['macro avg']['recall'], step=epoch)
        tf.summary.scalar(f'epoch_f1-score_macro', data=class_report['macro avg']['f1-score'], step=epoch)

        tf.summary.scalar(f'epoch_precision_weighted', data=class_report['weighted avg']['precision'], step=epoch)
        tf.summary.scalar(f'epoch_recall_weighted', data=class_report['weighted avg']['recall'], step=epoch)
        tf.summary.scalar(f'epoch_f1-score_weighted', data=class_report['weighted avg']['f1-score'], step=epoch)

        return class_report
