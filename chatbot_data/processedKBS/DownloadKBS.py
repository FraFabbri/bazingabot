
# coding: utf-8

# In[ ]:


import json, urllib3
import urllib.request
# Path
MainPath = 'C:/Users/Francesco/Desktop/chatbot/'
path = MainPath + 'chatbot_data/processedKBS/'
#API-key
key = 'INSERT_BABELNET_API_KEY' # babelnet API-key

# extract number of observations
number_of_items = 'http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/items_number_from?id=0&key=' + key
max_number = int(urllib.request.urlopen(number_of_items).read())
print(max_number)

interval = range(0,max_number, 5000)
done = ()
one_page_url = 'http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/items_from?id=%s&key='%x + key
page_string = urllib.request.urlopen(one_page_url).read()
page_for_json = json.loads(page_string)

# Download all the data
try:
    for idx, x in enumerate(interval):
        while True:
            try:
                page_for_json = None
                one_page_url = (
                    'http://151.100.179.26:8080/KnowledgeBaseServer/rest-api/items_from?id=%s&key='%x 
                                + key)                
                page_string = urllib.request.urlopen(one_page_url).read()
                page_for_json = json.loads(page_string)
            except Exception as e: 
                print()
                print(e)
                print()
                pass
            if page_for_json != [] and page_for_json != None:
                break
        if idx == 0:
            with open(path + 'kbs_entries.json', 'w') as outfile:   
                json.dump(page_for_json, outfile)
                outfile.close()
        else:
            with open(path + 'kbs_entries.json', 'r+') as f:   
                kbs_done = json.load(f)
                f.close()
            old_count = len(set([x['HASH'] for x in kbs_done]))
            kbs_done += page_for_json
            new_count = len(set([x['HASH'] for x in kbs_done]))
            if  old_count == new_count:
                print()
                print('Steady-state')
                print()
                raise StopIteration()
            else:
                with open(path + 'kbs_entries.json', 'r+') as f:   
                    json.dump(kbs_done, f)
                    f.close()

        if idx!=0:
            print(len(set([x['HASH'] for x in kbs_done])))
            print()
            print(x)
        print('-'*30)
        
except (StopIteration, KeyboardInterrupt):
    start = x
    done = (len(set([x['HASH'] for x in kbs_done])), start)
    print(done)
    pass

