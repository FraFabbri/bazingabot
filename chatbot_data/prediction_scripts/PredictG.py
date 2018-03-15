
# coding: utf-8

# In[1]:


import json, random, os, pickle
import numpy as np
import utilities.Utilities as Utilities



distinct_obs = 10**4
filename = 'general_' + str(int(distinct_obs/1000)) + 'k'
MainPath = 'C:/Users/Francesco/Desktop/chatbot/'
path = MainPath + 'chatbot_data/data/'

if filename + '.p' not in os.listdir(path):
    with open(path + 'general.json', 'r') as f: general = json.load(f)
    OneD = general
    # Arrange the data in a list of tuples
    SampleData = [(
        OneD[k]['question'].strip() + ' | ' + OneD[k]['domain'][0] + ' | ' + OneD[k]['relation'],
        OneD[k]['answer'].strip()
        ) 
        for k in OneD]

    dataset = list(set(SampleData))[:distinct_obs]
    random.seed(123)
    random.shuffle(dataset)
    print('Total number of distinct observations: ' + str(distinct_obs) + '\n')
    # Save dataset
    pickle.dump(dataset, open( path + filename + '.p', "wb"))
else:
    dataset = pickle.load(open(path + filename + '.p' , "rb" ))


# # train model until 0.95 is reached 


ModelSeq2Seq = Utilities.Seq2Seq()
# Process Data
ModelSeq2Seq.process_char(dataset = dataset, settings_path = path, settings_name = filename, setup = True)
ModelSeq2Seq.fill_data(dataset = dataset, conversation=False)


# Initialization - 100, 20
batch_size = 256
latent_dim = 256
ModelSeq2Seq.latent_dim = latent_dim
ModelSeq2Seq.define_model()    



# train model, epochs = 210
epochs = 200

path = MainPath + 'chatbot_data/models/Seq2Seq/general/'
if 'weightsSeq2Seq.keras' in os.listdir(path):
    ModelSeq2Seq.model.load_weights(path + 'weightsSeq2Seq.keras')
    ModelSeq2Seq.train_model(batch_size=batch_size, 
                             epochs=epochs, 
                             path = path, 
                             set_accuracy='no')
else:
    ModelSeq2Seq.train_model(batch_size=batch_size, 
                             epochs=epochs, 
                             path = path, 
                             set_accuracy='no')

