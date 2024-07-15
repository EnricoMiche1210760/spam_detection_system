import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU
from tensorflow.keras import Sequential

class EarlyStopping(tf.keras.callbacks.Callback):
    '''
    Early stopping callback to stop training when the validation accuracy is greater than 0.97
    '''
    def __init__(self, minimum_epochs=3):
        '''
        Initialization of the EarlyStopping class
        Parameters:
            minimum_epochs: int, minimum number of epochs before stopping
        '''
        self._minimum_epochs = minimum_epochs
    def on_epoch_end(self, epoch, logs):
        '''
        Function to stop training when the validation accuracy is greater than 0.97
        Parameters:
            epoch: int, current epoch
            logs: dict, logs of the training
        '''
        if logs['val_accuracy'] > 0.97 and epoch >= self._minimum_epochs:
            self.model.stop_training = True
            print('\nStop training at epoch:', epoch+1)

@tf.keras.saving.register_keras_serializable()
def weighted_binary_crossentropy(w0, w1):
    '''
    Function to compute the weighted binary crossentropy
    Parameters:
        w0: float, weight for class 0
        w1: float, weight for class 1
    '''
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        _w0 = tf.constant(w0, dtype=tf.float32)
        _w1 = tf.constant(w1, dtype=tf.float32)
        loss = -_w0 * y_true * tf.math.log(y_pred) - _w1 * (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return loss

def GRU_model(vocab_size, maxlen, activation='softmax'):
    '''
    Function to create a GRU model
    Parameters:
        vocab_size: int, size of the vocabulary
        maxlen: int, maximum length of the input
        activation: string, activation function for the output layer
    '''
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=256, input_length=maxlen))
    model.add(GRU(units=128))
    model.add(Dense(1, activation=activation))
    return model