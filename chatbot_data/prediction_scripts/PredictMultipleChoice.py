
# coding: utf-8

# In[ ]:


from keras.models import Sequential, Model
from keras.layers import Activation, TimeDistributed, Dense, RepeatVector, Embedding, Input
from keras.layers.recurrent import LSTM
from keras import backend as K
import json, random, os, pickle
import numpy as np
import utilities.Utilities as Utilities

MainPath = 'C:/Users/Francesco/Desktop/chatbot/'
path = MainPath + 'chatbot_data/data/'
with open (path + 'DataDizE.json', 'r') as f: DataDizE = json.load(f)

dataset = [(
    DataDizE[k]['question'],
    DataDizE[k]['domain'][0],
    DataDizE[k]['relation'],
    DataDizE[k]['answer']
    ) 
    for k in sorted(DataDizE.keys())]

dataset = sorted(set([(x + ' | ' + y + ' | ' + w, z) for x, y, w, z in dataset]))
dataset = dataset*10
random.seed(123)
random.shuffle(dataset)

LSTM_instance_model = Utilities.LSTMModel()
LSTM_instance_model.process_dataset(dataset, 1)

ModelPath = MainPath + 'chatbot_data/models/'

# Initialization NN
hidden_size = latent_dim = 128
batch_size = 512
embedding_length = 100
#K.clear_session()
epochs = 30
y_data = LSTM_instance_model.process_data_Y(LSTM_instance_model.Y,
                                            LSTM_instance_model.Y_max_len,
                                            LSTM_instance_model.Y_word_to_ix)

# Epochs: ~30 to 0.99
if 'PredictMC.keras' not in os.listdir(ModelPath):
    print('Build a new one...')
    LSTM_instance_model.model = Sequential()
    LSTM_instance_model.model.add(Embedding(LSTM_instance_model.X_vocab_len, 
                        embedding_length, 
                        input_length=LSTM_instance_model.X_max_len, 
                        mask_zero=True))
    LSTM_instance_model.model.add(LSTM(hidden_size))

    """
    Next, we will create the decoder network, which does the main job. First, we need to repeat the single vector outputted from the encoder network to obtain a sequence which has the same length with the output sequences. The rest is similar to the encoder network, except that the decoder will be more complicated, which we will have two or more hidden layers stacked up. For ones who are not familiar with Recurrent Neural Networks and how to create them using Keras, please refer to my previous post from the link in the beginning of this post.
    """

    LSTM_instance_model.model.add(RepeatVector(LSTM_instance_model.Y_max_len))
    LSTM_instance_model.model.add(LSTM(hidden_size, return_sequences=True))
    LSTM_instance_model.model.add(TimeDistributed(Dense(LSTM_instance_model.Y_vocab_len)))
    LSTM_instance_model.model.add(Activation('softmax'))
    LSTM_instance_model.model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop', metrics=['accuracy'])    
    # Train and Save
    try:
        LSTM_instance_model.train_model(epochs=epochs, 
                                        batch_size = batch_size)
        LSTM_instance_model.save('PredictMC', 'DataForPredictionMC', ModelPath)
    except KeyboardInterrupt:
        LSTM_instance_model.save('PredictMC', 'DataForPredictionMC', ModelPath)
else:
    print('Load model...')
    from keras.models import load_model
    LSTM_instance_model.model = load_model(ModelPath + 'PredictMC.keras')
    try:
        LSTM_instance_model.train_model(epochs=epochs, 
                                        batch_size = batch_size)
        LSTM_instance_model.save('PredictMC', 'DataForPredictionMC', ModelPath)
    except KeyboardInterrupt:
        LSTM_instance_model.save('PredictMC', 'DataForPredictionMC', ModelPath)

ratio = int(len(set(dataset))*.5)
test = sorted(set(dataset))[:ratio]
x_test, y_test = list(zip(*test))
# prediction
prediction = Utilities.test_model(x_test, ModelPath + 'DataForPredictionMC.p', ModelPath + 'PredictMC.keras')
tot_predicted = len(list(filter(lambda x: x[0] == x[1], list(zip(prediction,y_test)))))
with open(ModelPath + 'accuracy_multiple_choice_model.json', 'w') as f: json.dump({'accuracy': tot_predicted/len(y_test)}, f)

