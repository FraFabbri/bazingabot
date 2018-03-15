from keras.models import Sequential
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding
from keras.layers.recurrent import LSTM
from keras import backend as K
import json, random, os
import utilities.Utilities as Utilities

MainPath = 'C:/Users/Francesco/Desktop/chatbot/'
path = MainPath + 'chatbot_data/data/'
with open(path + 'DataDizYN.json', 'r') as f: DataDizYN = json.load(f)

# Select Num of observations used for training the Model
Data = [(
    DataDizYN[k]['question'],
    DataDizYN[k]['domain'][0],
    DataDizYN[k]['relation'],
    DataDizYN[k]['answer']
    ) 
    for k in sorted(DataDizYN.keys())]

Data = sorted(set([(x + ' | ' + y + ' | ' + w, z) for x, y, w, z in Data]))

# set model
Model = Utilities.LSTMModel()
Model.process_dataset(Data, 1)

# Initialization NN
hidden_size = 128
batch_size = 256
AtTime = len(Data)
embedding_length = 10**2

# Create NN-Architecture
epochs = 5
ModelPath = MainPath + 'chatbot_data/models/'

K.clear_session()
if 'PredictYN.keras' not in os.listdir(ModelPath):
    Model.model = Sequential()
    Model.model.add(Embedding(Model.X_vocab_len,
                              embedding_length,
                              input_length=Model.X_max_len,
                              mask_zero=True))

    Model.model.add(LSTM(hidden_size))
    Model.model.add(RepeatVector(Model.Y_max_len))
    Model.model.add(TimeDistributed(Dense(Model.Y_vocab_len)))
    Model.model.add(Activation('softmax'))

    Model.model.compile(loss='categorical_crossentropy',
                        optimizer='rmsprop',
                        metrics=['accuracy'])

    # Train and Save
    Model.train_model(batch_size=batch_size, epochs=epochs)
    Model.save('PredictYN', 'DataForPredictionYN', ModelPath)
else:
    from keras.models import load_model
    Model.model = load_model(ModelPath + 'PredictYN.keras')
    Model.train_model(batch_size=batch_size, epochs=epochs)

