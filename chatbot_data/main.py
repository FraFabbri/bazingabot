from modules.OneLib import *
import warnings

warnings.filterwarnings("ignore")
def main():
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

if __name__=='__main__':
    main()
