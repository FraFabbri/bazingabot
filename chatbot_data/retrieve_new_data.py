
# coding: utf-8

# In[147]:


import pandas as pd
import os, json, gzip

extraced_QAs = []
for conversation in os.listdir('conversations'):
    processed = pd.read_csv('conversations/' + conversation + '/' + 'processed_msg.csv', index_col=0)
    enriching = pd.read_csv('conversations/' + conversation + '/' + 'enriching_df.csv', index_col=0)
    new_q = [enriching.loc[idx].to_dict() for idx in enriching.index]
    count = 0
    for row in processed.index:
        if processed.loc[row]['bot_reply'] == '31q':
            new_q[count]['answer'] = processed.loc[row]['text']
            count +=1
    extraced_QAs += new_q

def retrieve_synset(text):
    """
    """
    import urllib.parse, urllib.request
    from io import BytesIO
    
    service_url = 'https://babelfy.io/v1/disambiguate'
    lang = 'EN'
    key  = '6d4850cd-b4ed-46ac-9bdc-5187a3a133eb'

    params = {'text' : text,
              'lang' : lang,
              'key'  : key}
    url = service_url + '?' + urllib.parse.urlencode(params)
    request =  urllib.request.Request(url)

    request.add_header('Accept-encoding', 'gzip')
    response = urllib.request.urlopen(request)
    if response.info().get('Content-Encoding') == 'gzip':
        buf = BytesIO( response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())    
    synsetId = [x['babelSynsetID'] for x in data if x['babelSynsetID'][-1] == 'n' or len(data) ==1]
    return synsetId

only_meaninigful = [dic for dic in extraced_QAs if "I don't know" not in dic['answer']]
for dic in only_meaninigful:
    dic['c2'] = retrieve_synset(dic['answer'])[0]
    del dic['id_telegram']

path = 'data/'
if 'new_data.json' not in os.listdir('.'):
    with open(path + 'new_data.json', 'w') as f: json.dump(only_meaninigful, f)
else:
    with open(path + 'new_data.json', 'r') as f: new_data = json.load(f)
    new_data += only_meaninigful
    with open(path + 'new_data.json', 'w') as f: json.dump(new_data, f)

