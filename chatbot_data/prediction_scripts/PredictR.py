
# coding: utf-8

from keras.models import Sequential, Model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input
from keras.layers.recurrent import LSTM
from keras import backend as K
import json, random, os, pickle
import numpy as np
import utilities.Utilities as Utilities

MainPath = 'C:/Users/Francesco/Desktop/chatbot/'
path = MainPath + 'chatbot_data/data/'
dataset1 = pickle.load(open(path + 'dataset1.p', 'rb'))
# 204527 obs
traindata = dataset1[:int(len(dataset1)*.8)]
testdata = dataset1[int(len(dataset1)*.8):]


LSTM_instance_model = Utilities.LSTMModel()
LSTM_instance_model.process_dataset(traindata, 0.5)


# Initialization NN
hidden_size = 128
batch_size = 256
embedding_length = 10**2

# Create NN-Architecture
epochs = 1
ModelPath = MainPath + 'chatbot_data/models/'


K.clear_session()
if 'PredictR.keras' not in os.listdir(ModelPath):
    print('Build a new one...')
    LSTM_instance_model.model = Sequential()
    LSTM_instance_model.model.add(Embedding(LSTM_instance_model.X_vocab_len,
                                            embedding_length,
                                            input_length=LSTM_instance_model.X_max_len,
                                            mask_zero=True))

    LSTM_instance_model.model.add(LSTM(hidden_size))
    LSTM_instance_model.model.add(RepeatVector(LSTM_instance_model.Y_max_len))
    LSTM_instance_model.model.add(TimeDistributed(Dense(LSTM_instance_model.Y_vocab_len)))
    # LSTM_instance_model.model.add(Dense(LSTM_instance_model.Y_vocab_len))
    LSTM_instance_model.model.add(Activation('softmax'))

    LSTM_instance_model.model.compile(loss='categorical_crossentropy',
                                      optimizer='rmsprop',
                                      metrics=['accuracy'])

    # Train and Save
    LSTM_instance_model.train_model(batch_size=batch_size, epochs=epochs)
    LSTM_instance_model.save('PredictR', 'DataForPredictionR', ModelPath)
else:
    print('Load model...')
    from keras.models import load_model
    LSTM_instance_model.model = load_model(ModelPath + 'PredictR.keras')
    LSTM_instance_model.train_model(batch_size=batch_size, epochs=epochs)
    LSTM_instance_model.save('PredictR', 'DataForPredictionR', ModelPath)


x_test, y_test = zip(*testdata)
prediction = Utilities.test_model(x_test, ModelPath + 'DataForPredictionR.p', ModelPath + 'PredictR.keras')
tot_predicted = len(list(filter(lambda x: x[0] == x[1], list(zip(prediction,y_test)))))
with open(ModelPath + 'accuracy_relation_model.json', 'w') as f: json.dump({'accuracy': tot_predicted/len(y_test)}, f)

