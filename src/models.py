import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Embedding,\
      LSTM, Dense, Dropout, GRU, Dropout, BatchNormalization
from tensorflow.keras import Sequential

class EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, minimum_epochs=3):
        self._minimum_epochs = minimum_epochs
    def on_epoch_end(self, epoch, logs):
        if logs['val_accuracy'] > 0.97 and epoch >= self._minimum_epochs:
            self.model.stop_training = True
            print('\nStop training at epoch:', epoch+1)

class BatchLogger(tf.keras.callbacks.Callback):
    def on_batch_end(self, batch_number, logs):
        if batch_number % 150 == 0:
            self.batch_loss.append(logs['loss'])
            self.batch_accuracy.append(logs['accuracy'])
    def on_train_begin(self, *args, **kwargs):
        self.batch_loss = []
        self.batch_accuracy = []


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

@tf.keras.saving.register_keras_serializable()
def weighted_binary_crossentropy(w0, w1):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-1e-7)
        _w0 = tf.constant(w0, dtype=tf.float32)
        _w1 = tf.constant(w1, dtype=tf.float32)
        loss = -_w0 * y_true * tf.math.log(y_pred) - _w1 * (1 - y_true) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return loss

def LSTM_CNN_model(vocab_size, maxlen, dense_units=1, activation='softmax'):
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=256, input_length=maxlen))
    model.add(Dropout(0.8))
    model.add(Conv1D(filters=64, kernel_size=3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(LSTM(units=64))
    model.add(Dropout(0.6))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(dense_units, activation=activation))
    return model

def GRU_model(vocab_size, maxlen, activation='softmax'):
    model = Sequential()
    model.add(Embedding(input_dim = vocab_size, output_dim=256, input_length=maxlen))
    model.add(GRU(units=128))
    model.add(Dense(1, activation=activation))
    return model