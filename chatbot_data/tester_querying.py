
import PredictSingleObs, sys

def main(obs):
	text = ' | '. join(obs.split('|')[:-1]).strip()
	text_to_predict_1 = text
	res1 = PredictSingleObs.main(text_to_predict_1, 'R')   
	relation = '_'.join(res1.split('_')[:-1])
	type_q = res1.split('_')[-1]
	text_to_predict_2 = text_to_predict_1 + ' | ' + relation  
	res2 = PredictSingleObs.main(text_to_predict_2, type_q)
	res = (obs, res1, res2.strip())
	print(res) 

if __name__=='__main__':
    sys.exit(main(sys.argv[1]))
