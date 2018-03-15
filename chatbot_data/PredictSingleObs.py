import sys

def main(arg1, arg2):
	import modules.Utilities as Utilities
	MainPath = "C:/Users/Francesco/Desktop/chatbot/"
	paths = { 
				'R': ("chatbot_data/models/DataForPredictionR.p", "chatbot_data/models/PredictR.keras"),
				'YN': ("chatbot_data/models/DataForPredictionYN.p", "chatbot_data/models/PredictYN.keras"),
				'E': ("chatbot_data/models/DataForPredictionMC.p", "chatbot_data/models/PredictMC.keras"), # E = MC
				'G': "chatbot_data/models/Seq2Seq/general/weightsSeq2Seq.keras"
			}

	msg, selection = arg1, arg2
	# Relation and YN
	if selection in ['R', 'YN', 'E']:

		data_path, model_path = MainPath + paths[selection][0], MainPath + paths[selection][1]
		predicted = Utilities.test_model(obs = [msg], 
									data_path = data_path, 
									model_path = model_path)[0]
		if 'PATTERN' in predicted:
			import json
			with open(MainPath + 'chatbot_data/data/PatternAnswer.json', 'r') as f: pattern_answer = json.load(f)
			if len(pattern_answer[predicted]) == 1:
				return pattern_answer[predicted][0]
			else:
				import random
				random_ans = random.sample(pattern_answer[predicted], 1)[0]
				return random_ans

		else:
			return predicted
	# Seq2Seq
	if selection == 'G':
		import pickle
		path =  MainPath + 'chatbot_data/data/'
		filename = 'general_10k'
		dataset = pickle.load( open(path + filename + '.p', "rb" ))
		ModelSeq2Seq = Utilities.Seq2Seq()

		# Process Data
		ModelSeq2Seq.process_char(dataset = dataset, settings_path = path, settings_name = filename, setup = True)
		ModelSeq2Seq.fill_data(dataset = dataset, conversation=False)

		# Initialization - 100, 20
		batch_size = 256
		latent_dim = 256
		ModelSeq2Seq.latent_dim = latent_dim
		ModelSeq2Seq.define_model()    

		model_path = MainPath + paths['G']
		# prediction
		ModelSeq2Seq.model.load_weights(model_path)
		predicted =  ModelSeq2Seq.test_model(random_selection = 'no', obs_to_predict = msg)
		return predicted

if __name__=='__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))