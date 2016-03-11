import keras
import pandas as pd
from keras.models import Sequential, Graph
from keras.layers.core import Dense, Dropout, Flatten, Merge, add_shared_layer
from keras.layers.convolutional import Convolution1D
from keras.utils import np_utils


def get_model_1(embedding_size):
    """Metrics after 15 epochs: 6s - loss: 0.3370 - acc: 0.8565 - val_loss: 0.3984 - val_acc: 0.8189"""
    n_classes = 2
    n_filter=3
    model = Sequential()
    model.add(Convolution1D(nb_filter=n_filter, filter_length=2, border_mode='valid',
                            input_shape=[embedding_size, 2]))
    model.add(Flatten())
    model.add(Dense(250, activation='relu', input_shape=[n_filter*embedding_size]))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return(model)


def prep_X(x_array):
    embedding_size = x_array.shape[1] / 2
    out = x_array.reshape([x_array.shape[0], embedding_size, 2], order='F')
    return embedding_size, out

if __name__ == "__main__":
    data = pd.read_csv('work/indico_pairs.csv')
    data = data.sample(frac=1) # shuffles rows
    embedding_size, X = prep_X(data.ix[:,2:].values) # first two columns are target and comment score
    target = np_utils.to_categorical(data.true_pair)
    batch_size = 128
    model = get_model_1(embedding_size)
    model.fit(X, target, batch_size=batch_size, nb_epoch=15,
              show_accuracy=True, verbose=1, validation_split=0.3)
    model.save_weights('work/relevance_model.h5', overwrite=True)
