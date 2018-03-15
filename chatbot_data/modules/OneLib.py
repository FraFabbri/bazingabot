
# coding: utf-8


from twx.botapi import TelegramBot, ReplyKeyboardMarkup, get_updates
import os, json, time
import pandas as pd

"""
Setup bot
"""
bot_api = '443605276:AAEspg0DAEc4s0TAmHSBKUDV-6EewfTVc_8'
bot = TelegramBot(bot_api)
bot.update_bot_info().wait()

MainPath = "C:/Users/Francesco/Desktop/chatbot/chatbot_FrancescoFabbri/"
ConversationsPath = MainPath + "chatbot_data/conversations/"
SettingsPath = MainPath + 'chatbot_data/settings/'
list_domains = pd.read_csv(MainPath + 'BabelDomains_full/domain_list.txt', sep = '\n', header=None)

"""
Build Legend
"""

col_names = ['code', 'code_name', 'bot_reply','topic', 'next']
df = pd.DataFrame(columns=[col_names])
# build df
df.loc['start'] = ['start', 'first_msg', '0q', None, '0']
df.loc[0] = [0, 'hi', '1q', None,'1']
df.loc[1] = [1, 'start', '2q', None,'2']
df.loc[2] = [2, 'topic', '3q-4q','topic', '3-4']
df.loc[3] = [3, 'enriching_querying', '31q', 'topic', '31']
df.loc[4] = [31, 'answer', '311q-312q', 'topic', '311-312']
df.loc[5] = [311, 'yes', '1q-3q-end', 'topic', '1-3-end']
df.loc[6] = [312, 'no', '3121q-3q', 'topic', '3']
df.loc[7] = [4, 'ask_me', '41q', 'topic', '41']
df.loc[8] = [41, 'question', '411q-412q', 'topic', '411-4']
df.loc[9] = [411, 'yes', '1q-4q-end', 'topic', '1-4-end']
df.to_csv(MainPath + 'chatbot_data/others/SummaryOperations.csv', index=None)

"""
Functions
"""

def welcome_0q(chat_id):

    msg = "Hey dude! I'm Bazinga, your friendly neighbourhood Spider-Bot!"
    keyboard = [
    ['Start']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg, reply_markup=reply_markup).wait()

def choose_topic_1q(chat_id):
    """
    """
    import random
    random5 = random.sample(list_domains[0].tolist(), 5)
    msg = "Here we are!\nSooooo...\nWhat do you want to talk about?"
    keyboard = [
    random5
    ]

    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg, reply_markup=reply_markup).wait()

def type_conversation_2q(chat_id):
    msg = "\nDo you want to ask (Querying) or tell (Enriching) me something?"
    keyboard = [
    ['Enriching', 'Querying']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg, reply_markup=reply_markup).wait()

def random_question_3q(dir_chat, chat_id, topic):
    import pickle, time, os, json, random
    path = MainPath + 'chatbot_data/data/'
    with open(path  + 'enriching_data.json', 'r') as f: enriching_data = json.load(f)

    original_topic = topic
    msg = "Let me think...\n"
    bot.send_message(chat_id, msg).wait()
    count = 0
    while True:
        lst = [x for x in enriching_data.keys() if topic in x]
        if len(lst) > 1:
            break
        else:
            if count < 1:
                msg = "I have no questions related to this topic, I'm going to change it"
                bot.send_message(chat_id, msg).wait()
                topic = random.sample([x.split('|')[0].strip() for x in enriching_data], 1)[0]
                count +=1

    
    # Select relation and question
    relation = random.sample(lst,1)[0]
    question = random.sample(enriching_data[relation].keys(), 1)[0]
    msg_dic = {
	    'id_telegram': chat_id,
	    'question': question,
	    'domain': topic,
	    'relation': relation.split('|')[1],
	    'c1': enriching_data[relation][question]['c1'],
	    'c2': 'NOT_YET'}

    if 'enriching_df.csv' not in os.listdir(dir_chat):
	    sortedCols = sorted(['id_telegram','question', 'domain','relation', 'c1', 'c2'])
	    df = pd.DataFrame(columns=[sortedCols])
	    sorted_values = list(zip(*sorted(msg_dic.items(), key = lambda x: x[0])))[1]
	    df.loc[0] = sorted_values
	    df.to_csv(dir_chat + '/enriching_df.csv')
    else:
	    sorted_values = list(zip(*sorted(msg_dic.items(), key = lambda x: x[0])))[1]
	    df = pd.DataFrame.from_csv(dir_chat + '/enriching_df.csv')
	    df.loc[len(df.index)] = sorted_values
	    df.to_csv(dir_chat + '/enriching_df.csv', index = True)

    if original_topic != topic:
        bot.send_message(chat_id, "Let's talk about %s"%topic.replace('_', ' ')).wait()
    bot.send_message(chat_id, question + '?').wait()
    return topic

def check_31q(chat_id):
    msg = "Looking at the question and the answer, can I learn from our conversation?"
    keyboard = [
    ['Yes', 'No']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg, reply_markup=reply_markup).wait()

def store_and_go_311q(dir_chat, chat_id, text):
    msg = 'Let me store this new info...'
    question = 'Now, how do you want to proceed?'
    keyboard = [
    ['New conversation', 'New question with same topic', 'Bye']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg).wait()
    bot.send_message(chat_id, question, reply_markup=reply_markup).wait()

def fix_issues_312q(chat_id):
    msg = "What do we need to change?"
    keyboard = [
    ['question', 'answer']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg, reply_markup=reply_markup).wait()

def new_answer_3121q(chat_id):
    msg = "Ok, write now your answer"
    bot.send_message(chat_id, msg).wait()

def ask_me_4q(chat_id):
    msg = "Ask me something!\n"
    bot.send_message(chat_id, msg).wait()

def bye(chat_id):
    msg = "Bye! See you soon...\n"
    bot.send_message(chat_id, msg).wait()    

def new_conversation_END(chat_id):
    msg = 'Here we go again!'
    bot.send_message(chat_id, msg).wait()  


def give_answer_41q(chat_id, msg, dict_msg):
    import PredictSingleObs, string

    if msg[-1] == '?':
        msg = msg[:-1]
    specials = set(string.punctuation) - {'?', '.', '-', "'", '&', ',', "'"}
    specials.update({'チ',
                     '乌',
                     '玄',
                     '米',
                     '茶',
                     '药',
                     '�'})
    for char in specials:
        msg = msg.replace(char, '')

    text_to_predict_1 = msg + ' | ' + dict_msg['topic']
    res1 = PredictSingleObs.main(text_to_predict_1, 'R')
    relation = '_'.join(res1.split('_')[:-1])
    type_q = res1.split('_')[-1]
    text_to_predict_2 = text_to_predict_1 + ' | ' + relation  
    res2 = PredictSingleObs.main(text_to_predict_2, type_q)
    bot.send_message(chat_id, res2).wait()
    
    time.sleep(1)
    msg2 = "Is my answer reasonable?"
    keyboard = [
    ['Yes', 'No']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg2, reply_markup=reply_markup).wait()


def keep_going_411q(chat_id):
    msg = 'Ok...'
    question = 'Now, how do you want to proceed?'
    keyboard = [
    ['New conversation', 'New question with same topic', 'Bye']
    ]
    reply_markup = ReplyKeyboardMarkup.create(keyboard, one_time_keyboard = True)
    bot.send_message(chat_id, msg).wait()
    bot.send_message(chat_id, question, reply_markup=reply_markup).wait()

def change_question_412q(chat_id):
    msg = "Ok, let's try again...\nAsk me something else of the same topic!"
    bot.send_message(chat_id, msg).wait()




def process_start(dict_msg, chat_id,dir_chat):
    """
    """
    welcome_0q(chat_id)
    sortedCols = sorted(['id_telegram','text','code','bot_reply','topic'])
    df = pd.DataFrame(columns=[sortedCols])
    processed_msg = {
        'id_telegram': dict_msg['id_telegram'],
        'text': dict_msg['text'],
        'code': 0,
        'bot_reply': '0q',
        'topic': None
    }
    sorted_values = list(zip(*sorted(processed_msg.items(), key = lambda x: x[0])))[1]
    df.loc[len(df.index)] = sorted_values
    df.to_csv(dir_chat + '/processed_msg.csv')

def process_END(dir_chat, chat_id, dict_msg):
    new_conversation_END(chat_id)
    welcome_0q(chat_id)
    processed_msg = {
        'id_telegram': dict_msg['id_telegram'],
        'text': dict_msg['text'],
        'code': 0,
        'bot_reply': '0q',
        'topic': None
    }
    return processed_msg    

def process_0(dir_chat, chat_id, dict_msg):
    """
    """
    if dict_msg['text'].lower() == 'start' or dict_msg['text'].lower() == 'new conversation':
        choose_topic_1q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 1,
            'bot_reply': '1q',
            'topic': None
        }
        return processed_msg

def process_1(dir_chat, chat_id, dict_msg):
    """
    """
    if dict_msg['text'].lower() in list_domains[0].apply(lambda x: x.lower()).values:
        type_conversation_2q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 2,
            'bot_reply': '2q',
            'topic': dict_msg['text']
        }
        return processed_msg

def process_2(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    if txt =='enriching':
        topic = random_question_3q(dir_chat, chat_id,dict_msg['topic'])
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 3,
            'bot_reply': '3q',
            'topic': topic
        }
        return processed_msg
    elif txt == 'querying':
        ask_me_4q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 4,
            'bot_reply': '4q',
            'topic': dict_msg['topic']
        }
        return processed_msg

def process_3(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    check_31q(chat_id)
    processed_msg = {
        'id_telegram': dict_msg['id_telegram'],
        'text': dict_msg['text'],
        'code': 31,
        'bot_reply': '31q',
        'topic': dict_msg['topic']
    }
    return processed_msg

def process_31(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    if txt == 'yes':
        store_and_go_311q(dir_chat, chat_id, dict_msg['text'])
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 311,
            'bot_reply': '311q',
            'topic': dict_msg['topic']
        }
        return processed_msg
    if txt == 'no':
        fix_issues_312q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 312,
            'bot_reply': '312q',
            'topic': dict_msg['topic']
        }
        return processed_msg

def process_311(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    if txt == 'new conversation':
        choose_topic_1q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 1,
            'bot_reply': '1q',
            'topic': None
        }
        return processed_msg
    if txt == 'new question with same topic':
        topic = random_question_3q(dir_chat, chat_id, dict_msg['topic'])
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 3,
            'bot_reply': '3q',
            'topic': topic
        }
        return processed_msg
    if txt == 'bye':
        bye(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 'END',
            'bot_reply': 'bye',
            'topic': None
        }
        return processed_msg                

def process_312(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    if txt == 'answer':
        new_answer_3121q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 3,
            'bot_reply': '3121q',
            'topic': dict_msg['topic']
        }
        return processed_msg
    if txt == 'question':
        topic = random_question_3q(dir_chat, chat_id, dict_msg['topic'])
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 3,
            'bot_reply': '3q',
            'topic': topic
        }
        return processed_msg   


def process_4(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    ### Process Question and give answer with msg
    give_answer_41q(chat_id, txt, dict_msg)
    # store apart Q&A
    
    processed_msg = {
        'id_telegram': dict_msg['id_telegram'],
        'text': dict_msg['text'],
        'code': 41,
        'bot_reply': '41q',
        'topic': dict_msg['topic']
    }
    return processed_msg

def process_41(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    if txt == 'yes':
        keep_going_411q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 411,
            'bot_reply': '411q',
            'topic': dict_msg['topic']
        }
        return processed_msg
    if txt == 'no':
        change_question_412q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 4,
            'bot_reply': '412q',
            'topic': dict_msg['topic']
        }
        return processed_msg

def process_411(dir_chat, chat_id, dict_msg):
    """
    """
    txt = dict_msg['text'].lower().strip()
    if txt == 'new conversation':
        choose_topic_1q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 1,
            'bot_reply': '1q',
            'topic': None
        }
        return processed_msg
    if txt == 'new question with same topic':
        ask_me_4q(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 4,
            'bot_reply': '4q',
            'topic': dict_msg['topic']
        }
        return processed_msg
    if txt == 'bye':
        bye(chat_id)
        processed_msg = {
            'id_telegram': dict_msg['id_telegram'],
            'text': dict_msg['text'],
            'code': 'END',
            'bot_reply': 'bye',
            'topic': None
        }
        return processed_msg    


def process_msg(dir_chat,chat_id,dict_msg):
    functions = {}
    functions["0"] = process_0
    functions["1"] = process_1
    functions["2"] = process_2
    functions["3"] = process_3
    functions["4"] = process_4
    functions["31"] = process_31
    functions["311"] = process_311
    functions["312"] = process_312
    functions["41"] = process_41
    functions["411"] = process_411
    functions['END'] = process_END
    try:
        df = pd.DataFrame.from_csv(dir_chat + '/processed_msg.csv')
        code_to_process = df.loc[df.index[-1]]['code']
        if df.loc[len(df.index)-1]['topic'] != None:
            dict_msg['topic'] = df.loc[len(df.index)-1]['topic']
        processed_msg = functions[str(code_to_process)](dir_chat = dir_chat,chat_id = chat_id, dict_msg = dict_msg)
        sorted_values = list(zip(*sorted(processed_msg.items(), key = lambda x: x[0])))[1]
        df.loc[len(df.index)] = sorted_values
        df.to_csv(dir_chat + '/processed_msg.csv', index = True)
    except IOError:
        process_start(dir_chat = dir_chat, dict_msg = dict_msg, chat_id = chat_id)



def update_last_ID(ID):
    with open(SettingsPath + 'last_id.json', 'r+') as out:    
        json.dump({'id': ID},out)
        out.close()
        
def get_last_ID():
    try:
        with open(SettingsPath + '/' + 'last_id.json', 'r+') as out:
            last_id = json.load(out)['id']
            out.close()
        return last_id
    except IOError:
        return 0

"""
Main Functions
"""


def call_bot():
    last_id = get_last_ID() + 1
    updates = bot.get_updates(offset = last_id, limit = 100).wait()
    temporary_log = [
        {'update_id': update.update_id,
         'sender_id': update.message.sender.id,
         'text': update.message.text,
         'chat_id': update.message.chat.id}
        for update in updates
    ]
    try:
        update_last_ID(max([res['update_id'] for res in temporary_log]))
    except ValueError:
        pass
    return temporary_log

def send_msg(temp_log):
    if temp_log != []:
        for d in temp_log:
            name_folder = str(d['sender_id']) + '_' + str(d['chat_id'])
            try:
                df = pd.read_csv(ConversationsPath + 
                                 '/' + name_folder + 
                                 '/' + 'history' + '.csv',index_col=0)
                last_id = df.index[-1]
                df.loc[last_id + 1] = [d['update_id'], d['text'], 'waiting']
                df.to_csv(ConversationsPath + 
                                 '/' + name_folder + 
                                 '/' + 'history' + '.csv')
            except IOError:
                os.mkdir(ConversationsPath + 
                             '/' + name_folder)
                df = pd.DataFrame(columns=[['id_telegram', 'text', 'status']])
                df.loc[0] = [d['update_id'], d['text'], 'waiting']
                df.to_csv(ConversationsPath + 
                                 '/' + name_folder + 
                                 '/' + 'history' + '.csv')


def main_chat():
    try:
        while True:
            try:
                temporary_log = call_bot()
                send_msg(temporary_log)
            except KeyboardInterrupt:
                break    
            lst_chat = os.listdir(ConversationsPath)
            if lst_chat != []:
                for chat_folder in lst_chat:
                    dir_chat = ConversationsPath + chat_folder
                    chat_id = chat_folder.split('_')[1]
                    history_msg = pd.DataFrame.from_csv(dir_chat + '/history.csv')
                    waiting_msg = history_msg[history_msg.status == 'waiting']
                    if len(waiting_msg) >= 1:
                        dict_msg = waiting_msg.loc[waiting_msg.index[0]].to_dict()
                        process_msg(dir_chat=dir_chat, chat_id=chat_id, dict_msg=dict_msg)
                        history_msg.ix[waiting_msg.index[0], 'status'] = 'done'
                        history_msg.to_csv(dir_chat + '/history.csv')
                    else:
                        time.sleep(2)
            else:
                time.sleep(2)
    except KeyboardInterrupt:
        pass
