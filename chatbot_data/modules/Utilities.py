
# coding: utf-8



class LSTMModel():
    """
    """
    @staticmethod
    def process_data_Y(word_sentences, max_len, word_to_ix):
        
        """
        """
        import numpy as np

        # Vectorization of each element in each sequence
        sequences = np.zeros((len(word_sentences), max_len, len(word_to_ix)), dtype = 'bool')
        for i, sentence in enumerate(word_sentences):
            for j, word in enumerate(sentence):
                sequences[i, j, word] = True
        return sequences
    
    def process_dataset(self, SampleData = None, fraction = 1):
        import numpy as np
        import random
        from keras.preprocessing.sequence import pad_sequences
        from nltk import FreqDist

        # Prepare Data
        X, Y = list(zip(*SampleData))
        X = [ x.split()[::-1] for x in X]
        Y = [y.split()[::-1] for y in Y]
        dist = FreqDist(np.hstack(X))
        SortedVocab = sorted(dist.items(), key=lambda x: x[1], reverse=True)
        N = 100.0/fraction
        print('Using only the Top-%s%% of the Words'%(fraction*100))
        vocabX = list(dist.keys())[:int(len(SortedVocab)*fraction)] 
        vocabY = []
        for y in Y: vocabY += y
        vocabY = list(set(vocabY))
        # X-data
        self.X_ix_to_word = list(vocabX)
        self.X_ix_to_word.insert(0, 'ZERO')
        self.X_ix_to_word.append('UNK')
        self.X_word_to_ix = {word:ix for ix, word in enumerate(self.X_ix_to_word)}
        # Converting each word to its index value
        for i, question in enumerate(X):
            for j, word in enumerate(question):
                if word in self.X_word_to_ix:
                    X[i][j] = self.X_word_to_ix[word]
                else:
                    X[i][j] = self.X_word_to_ix['UNK']
        # Y-data
        self.Y_ix_to_word = sorted(vocabY)
        self.Y_ix_to_word.insert(0, 'ZERO')
        self.Y_word_to_ix = {word:ix for ix, word in enumerate(self.Y_ix_to_word)}
        for i, sentence in enumerate(Y):
            for j, word in enumerate(sentence): Y[i][j] = self.Y_word_to_ix[word]

        # Vocabularies Lenghts
        self.X_vocab_len = len(vocabX) + 2 # =  + 'UNK' + 'ZERO'
        self.Y_vocab_len = len(vocabY) + 1 # = + 'ZERO'
        # Finding the length of the longest sequence
        self.X_max_len = max([len(sentence) for sentence in X])
        self.Y_max_len = max([len(sentence) for sentence in Y])

        # Padding zeros to make all sequences have a same length with the longest one
        X = pad_sequences(X, maxlen=self.X_max_len, dtype='int32')
        Y = pad_sequences(Y, maxlen=self.Y_max_len, dtype='int32')
        # Shuffling the training data
        self.X = X
        self.Y = Y

    def save(self, model_name, data_name, output_path):
        """
        """
        import pickle
        self.model.save(output_path + model_name + '.keras')
        DataForPrediction = {'X_word_to_ix': self.X_word_to_ix, 
                              'Y_ix_to_word': self.Y_ix_to_word, 
                              'X_max_len': self.X_max_len}
        pickle.dump(DataForPrediction, open(output_path + data_name + '.p', "wb"))
    
    def train_model(self, epochs=1, batch_size=64):
        """
        """

        try:
            y_sequences = self.process_data_Y(self.Y, 
                                              self.Y_max_len, 
                                              self.Y_word_to_ix)

            self.model.fit(self.X, 
                           y_sequences, 
                           batch_size=batch_size, 
                           epochs=epochs,
                           verbose=1)
        except Exception as e:
            if type(e) == NameError:
                print(e)
                print('Check that you defined all the data-variables')
            elif type(e) == AttributeError:
                print('Have you defined the model?')
            else:
                print(e)
            # raise 'No model defined'


def test_model(obs, data_path, model_path):
    """
    """
    """
    Load a sequential model saved in the file pointed by model_file_path

    :param model_file_path: the path to the file that has to be loaded
    :return: a sequential model loaded from the file
    """

    import numpy as np
    from keras.preprocessing.sequence import pad_sequences
    import pickle, keras

    DataForPrediction = pickle.load(open(data_path, 'rb'))
    (X_word_to_ix, 
     Y_ix_to_word, 
     X_max_len, 
     model) = (
        DataForPrediction['X_word_to_ix'], 
        DataForPrediction['Y_ix_to_word'],
        DataForPrediction['X_max_len'],
        keras.models.load_model(model_path)
    )

    Xtest = [x.split()[::-1] for x in obs]
    for idx_obs, single_obs in enumerate(Xtest):
        for j, word in enumerate(single_obs):
            if word in X_word_to_ix:
                single_obs[j] = X_word_to_ix[word]
            else:
                single_obs[j] = X_word_to_ix['UNK']
        Xtest[idx_obs] = single_obs
    Xtest = pad_sequences(Xtest, maxlen=X_max_len, dtype='int32')
    prediction = np.argmax(model.predict(Xtest), axis=2)
    return [Y_ix_to_word[one_pred[0]] for one_pred in prediction]

# test
#path = 'C:/Users/Francesco/Desktop/chatbot/chatbot_data/models/'
#test = 'Can Wolverhampton Art Gallery be found in Wolverhampton | Geography and places'
#test_model(test, data_path, model_path)



class Seq2Seq():
    """
    """

    def process_char(self, 
                     dataset,
                     settings_path,
                     settings_name, 
                     setup):
        """
        """
        import random, pickle,os

        if setup == True:            
            # Vectorize the data
            input_X, output_Y = list(zip(*[(input_text, '\t' + target_text + '\n')
                                                     for (input_text, target_text) in dataset]))

            #self.indices = np.arange(len(self.input_X))
            self.encoder_vocab = set()
            self.decoder_vocab = set()
            for obs in zip(input_X, output_Y):
                self.encoder_vocab.update(set(obs[0]))
                self.decoder_vocab.update(set(obs[1]))

            (self.encoder_max_len,
             self.decoder_max_len,
             self.encoder_vocab_len,
             self.decoder_vocab_len) = (max([len(txt) for txt in input_X]),
                                        max([len(txt) for txt in output_Y]),
                                        len(self.encoder_vocab),
                                        len(self.decoder_vocab))

            self.encoder_ix_to_word = sorted(self.encoder_vocab)
            self.encoder_word_to_ix = {word: ix for ix, word in enumerate(self.encoder_ix_to_word)}
            self.decoder_ix_to_word = sorted(self.decoder_vocab)
            self.decoder_word_to_ix = {word: ix for ix, word in enumerate(self.decoder_ix_to_word)}

            settings = {
                        'encoder_vocab': self.encoder_vocab,
                        'decoder_vocab': self.decoder_vocab,
                        'encoder_max_len': self.encoder_max_len,
                        'decoder_max_len': self.decoder_max_len,
                        'encoder_ix_to_word': self.encoder_ix_to_word,
                        'decoder_ix_to_word': self.decoder_ix_to_word,
                        }
            pickle.dump(settings, open(settings_path + 'settings_' +settings_name + '.p' , "wb"))

        else:
            settings = pickle.load(open(settings_path + 'settings_' +settings_name + '.p', "rb" ))
            #self.indices = np.arange(len(self.input_X))
            self.encoder_vocab = settings['encoder_vocab']
            self.decoder_vocab = settings['decoder_vocab']

            (self.encoder_max_len,
             self.decoder_max_len,
             self.encoder_vocab_len,
             self.decoder_vocab_len) = (settings['encoder_max_len'],
                                        settings['decoder_max_len'],
                                        len(self.encoder_vocab),
                                        len(self.decoder_vocab))

            self.encoder_ix_to_word = settings['encoder_ix_to_word']
            self.encoder_word_to_ix = {word: ix for ix, word in enumerate(self.encoder_ix_to_word)}
            self.decoder_ix_to_word = settings['encoder_ix_to_word']
            self.decoder_word_to_ix = {word: ix for ix, word in enumerate(self.decoder_ix_to_word)}
        

    def fill_data(self, 
                  dataset, conversation):
        """
        """
        import pandas as pd
        import numpy as np
        import random

        if conversation == False:
                dataset = 5*dataset

        random.seed(123)
        random.shuffle(dataset)

        # Vectorize the data
        self.input_X, self.output_Y = list(zip(*[(input_text, '\t' + target_text + '\n')
                                                 for (input_text, target_text) in dataset]))
        # Empty matrices
        self.encoder_input_data = np.zeros(
            (len(self.input_X), self.encoder_max_len, self.encoder_vocab_len),
            dtype='bool')
        self.decoder_input_data = np.zeros(
            (len(self.input_X), self.decoder_max_len, self.decoder_vocab_len),
            dtype='bool')
        self.decoder_target_data = np.zeros(
            (len(self.input_X), self.decoder_max_len, self.decoder_vocab_len),
            dtype='bool')

        # Fill the empty matricies
        for idx_obs, obs in enumerate(self.input_X):
            for idx_char, char in enumerate(obs):
                self.encoder_input_data[idx_obs, idx_char, self.encoder_word_to_ix[char]] = True
        for idx_obs, obs in enumerate(self.output_Y):
            for idx_char, char in enumerate(obs):
                self.decoder_input_data[idx_obs, idx_char, self.decoder_word_to_ix[char]] = True
                if idx_char > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    self.decoder_target_data[idx_obs, idx_char - 1, self.decoder_word_to_ix[char]] = True


    def define_model(self,
                     activation='softmax',
                     loss='categorical_crossentropy',
                     optimizer='rmsprop'):
        """
        """
        from keras.models import Model
        from keras.layers import Dense, Input
        from keras.layers.recurrent import LSTM
        from keras import backend as K
        K.clear_session()
        # Input sequence
        self._encoder_inputs = Input(shape=(None, self.encoder_vocab_len)) 
        encoder = LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(self._encoder_inputs)
        # Keeping the states.
        self._encoder_states = [state_h, state_c]
        # `encoder_states` as initial state.
        self._decoder_inputs = Input(shape=(None, self.decoder_vocab_len))
        # The decoder returns a full output sequences and the internal states
        self._decoder_lstm = LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = self._decoder_lstm(self._decoder_inputs,
                                                   initial_state=self._encoder_states)
        self._decoder_dense = Dense(self.decoder_vocab_len, activation=activation)
        decoder_outputs = self._decoder_dense(decoder_outputs)
        # [`encoder_input_data`, `decoder_input_data`] into `decoder_target_data`
        self.model = Model([self._encoder_inputs, self._decoder_inputs], decoder_outputs)
        self.model.compile(optimizer=optimizer, loss=loss)





    def test_model(self,
                   random_selection='yes',
                   N=2,
                   obs_to_predict=None):
        """
        """
        from keras.models import Model
        from keras.layers import  Input

        import numpy as np
        import random
        # Define model for prediction
        encoder_model = Model(self._encoder_inputs, self._encoder_states)
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = self._decoder_lstm(
            self._decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = self._decoder_dense(decoder_outputs)
        decoder_model = Model(
            [self._decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)


        def decode_sequence(input_seq):
            """
            """
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)
            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, self.decoder_vocab_len))
            # starter character.
            target_seq[0, 0, self.decoder_word_to_ix['\t']] = 1.
            # loop for extracting prediction of tokens
            stop_condition = False
            decoded_sentence = ''
            while True:
                output_tokens, h, c = decoder_model.predict(
                    [target_seq] + states_value)
                # Token with max prob
                sampled_token_index = np.argmax(output_tokens[0, -1, :])
                sampled_char = self.decoder_ix_to_word[sampled_token_index]
                decoded_sentence += sampled_char
                # Exit condition
                if (sampled_char == '\n' or
                            len(decoded_sentence) > self.decoder_max_len):
                    break
                # Update the target sequence
                target_seq = np.zeros((1, 1, self.decoder_vocab_len))
                target_seq[0, 0, sampled_token_index] = 1.
                # Update states
                states_value = [h, c]

            return decoded_sentence

        if random_selection == 'yes' and obs_to_predict == None:
            sample_data = random.sample(range(len(self.input_X)), N)

            for idx in sample_data:
                # generate randInt
                new_input_seq = self.encoder_input_data[idx: idx + 1]
                decoded_sentence = decode_sequence(new_input_seq)
                print('-')
                print('Input sentence:', self.input_X[idx])
                print('Decoded sentence:', decoded_sentence.strip())
                print('Expected sentence:', self.output_Y[idx].strip())
            print('\n')
        else:
            obs_array = np.zeros((1,
                                  self.encoder_max_len,
                                  self.encoder_vocab_len))
            for idx_char, char in enumerate(obs_to_predict):
                obs_array[0, idx_char, self.encoder_word_to_ix[char]] = 1.
            decoded_sentence = decode_sequence(obs_array)
            return decoded_sentence

    def train_model(self,
                    epochs=1    ,
                    batch_size=128,
                    path=None,
                    set_accuracy = 'no',
                    save = 'yes'):
        try:
            if set_accuracy == 'yes':
                self.compute_accuracy(10, save = 'no')
                while self.accuracy < 0.95:
                    self.model.fit(
                        [self.encoder_input_data, self.decoder_input_data], 
                        self.decoder_target_data,
                        batch_size=batch_size,
                        epochs=1)
                    self.test_model(N=2)   
                    if save == 'yes':
                        self.model.save_weights(path + 'weightsSeq2Seq.keras')
                    print('\nComputing accuracy...')
                    self.compute_accuracy(10, save = 'yes')
            else:
                for _ in range(epochs):
                    self.model.fit(
                        [self.encoder_input_data, self.decoder_input_data], 
                        self.decoder_target_data,
                        batch_size=batch_size,
                        epochs=1)
                    self.test_model(N=2)
                if save =='yes':   
                    self.model.save_weights(path + 'weightsSeq2Seq.keras')
                self.compute_accuracy(50, save = 'yes')

            
        except KeyboardInterrupt:
            print('\nSaved weights\n')
            self.model.save_weights(path + 'weightsSeq2Seq.keras')
            print('\nComputing accuracy...')
            self.compute_accuracy(10, save = 'no')


    def compute_accuracy(self, N, save):
        count = 0
        import random
        sample_data = random.sample(range(len(self.input_X)), N)
        for idx in sample_data:
            prediction = self.test_model(random_selection = 'no',
                                         obs_to_predict = self.input_X[idx])
            if prediction.strip() == self.output_Y[idx].strip():
                count +=1
        self.accuracy =  count/N
        if save == 'yes':
            # save to add
            print('MODIFY THE CODE!')
        # display results
        print('Accuracy: %s\n'%self.accuracy)
         
            
