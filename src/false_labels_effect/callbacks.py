import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import Callback


class class_report_cb(Callback):
    """
    Calculates and logs classification report for validatiot

    Parameters
    ----------

    Returns
    -------
    """

    def on_epoch_end(self, epoch, logs=None):
        print(self.model)
    #     y_pred = self.model.predict(self.x_val)
    #     report = classification_report(self.y_val, y_pred)
    #     logs['classification_report'] = report
    #     print(report)

        return
