def extract_main_sense(synset_id, key):
    """
    """
    import gzip , urllib, urllib.parse, json
    from urllib.request import urlopen, Request
    from io import BytesIO
    service_url = 'https://babelnet.io/v4/getSynset'
    id_synset = synset_id
    params = {'id' : id_synset,'key'  :  key}
    url = service_url + '?' + urllib.parse.urlencode(params)
    request = Request(url)
    request.add_header('Accept-encoding', 'gzip')
    response = urlopen(request)

    if response.info().get('Content-Encoding') == 'gzip':
        buf = BytesIO(response.read())
        f = gzip.GzipFile(fileobj=buf)
        data = json.loads(f.read())
    try:
        return data['mainSense']
    except KeyError:
        return 'CHANGE_KEY_BABELNET'

def look_for_bigger(string, lst, idx, lst_concepts, first):
    """
    """
    if string in lst_concepts:
        return 'UNK'
    else:
        idx +=1
        if idx == len(lst):
            return first
        string = string + ' ' + lst[idx]
        if len(string.split()) == len(lst):
            return first
        if string in lst_concepts:
            return string
        else:
            return look_for_bigger(string, lst, idx, lst_concepts, first) 

def extract_question_pattern(question, dict_concepts, synset_ids):
    import nltk
    tokenized_sentence = nltk.word_tokenize(question)
    new_sentence = []
    for idx, token in enumerate(tokenized_sentence):
        new_token = look_for_bigger(token, tokenized_sentence, idx, dict_concepts.values(), first =token)
        if new_sentence != []:
            if new_token in new_sentence[-1]:
                pass
            else:
                new_sentence.append(new_token)
        else:
            new_sentence.append(new_token)
    without_concepts = []
    for word in new_sentence:
        if word in dict_concepts.values():
            without_concepts.append('UNK')
        else:
            without_concepts.append(word)
    return (' '.join(without_concepts), synset_ids)


def build_enriching_tree(onlyQ, selected_synset):
    enriching_tree = {}
    for (q, id1, id2, domain, relation) in onlyQ:
        res = extract_question_pattern(question=q,
            synset_ids=[id1, id2], 
            dict_concepts = selected_synset) 

        q_parsed = res[0][:-1].strip()
        if q_parsed.count('UNK') == 1:
            if domain not in enriching_tree:
                enriching_tree[domain] = {}
                enriching_tree[domain]['synset_ids'] = set(res[1])
                enriching_tree[domain][relation] = set([q_parsed])
            else:
                enriching_tree[domain]['synset_ids'].update(set(res[1]))
                if relation not in enriching_tree[domain]:
                    enriching_tree[domain][relation] = set([q_parsed])
                else:
                    if q_parsed not in enriching_tree[domain][relation]:
                        enriching_tree[domain][relation].add(q_parsed)

    return enriching_tree

def generate_random_question(enriching_tree, domain, selected):
    import random
    one_domain = enriching_tree[domain]
    all_relations = list(set(one_domain.keys()) - set(['synset_ids']))
    random_relation = random.sample(all_relations,1)[0]
    if len(one_domain[random_relation]) == 1:
        random_question = list(one_domain[random_relation])[0]
    else:
        random_question = random.sample(list(one_domain[random_relation]), 1)[0]
    random_concept = random.sample(list(one_domain['synset_ids']), 1)[0]
    custom_q = random_question.replace('UNK', selected[random_concept])
    return (custom_q, random_relation, random_concept)