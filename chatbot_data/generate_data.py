
# coding: utf-8

# In[11]:


import pickle, json, random, PredictSingleObs
import pandas as pd

path = 'data/'
dataset_generic = pickle.load(open(path + 'general_10k' + '.p' , "rb" ))
with open (path + 'DataDizE.json', 'r') as f: DataDizE = json.load(f)
with open(path + 'DataDizYN.json', 'r') as f: DataDizYN = json.load(f)
with open(path + 'PatternAnswer.json', 'r') as f: PatternAnswer = json.load(f)

DataDizYN.update(DataDizE)

random.seed(123)
data = [(
    DataDizYN[k]['question'],
    DataDizYN[k]['domain'][0],
    DataDizYN[k]['relation'],
    DataDizYN[k]['answer']
    ) 
    for k in DataDizYN]

data = sorted(set([(x + ' | ' + y + ' | ' + w, z) for x, y, w, z in data]))

data = data + dataset_generic

print('Total number of distinct observations: %s'%len(data))

def main():
    while True:
        random_selected = random.sample(data, 5)
        for s in random_selected:
            print(s)
            if 'PATTERN' in s[-1]:
                print(PatternAnswer[s[-1]])
            print()
        stop = input()
        if len(stop) > 0:
            break

if __name__=='__main__':
    main()

