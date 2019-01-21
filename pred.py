import torch as t
import numpy as np
import cPickle
import sys,os
import argparse
import torch.utils.data as tdata
import util.utils as utils
from tqdm import tqdm
from models.HEER import HEER
from models.DistMult import DistMult

def parse_args():
	'''
	Parses the heer arguments.
	'''
	parser = argparse.ArgumentParser(description="Run heer.")

	parser.add_argument('--model', nargs='?', default='HEER',
						help='Which specific model to run')

	parser.add_argument('--more-param', nargs='?', default='None',
	                    help='customized parameter setting')

	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
	                    help='Input graph path')

	parser.add_argument('--gpu', nargs='?', default='0',
	                    help='Embeddings path')

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.')
	
	parser.add_argument('--batch-size', type=int, default=50,
	                    help='Batch size. Default is 50.')

	parser.add_argument('--window-size', type=int, default=1,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--pre-train-path', type=str, default='',
                    	help='embedding initialization')

	parser.add_argument('--build-graph', type=bool, default=False,
                    	help='heterogeneous information network construction')

	parser.add_argument('--graph-name', type=str, default='',
                    	help='prefix of dumped data')
	parser.add_argument('--data-dir', type=str, default='',
                    	help='data directory')
	parser.add_argument('--model-dir', type=str, default='',
                    	help='model directory')
	parser.add_argument('--test-dir', type=str, default='',
                    	help='test directory')

	parser.add_argument('--iter', default=500, type=int,
                      help='Number of epochs in SGD')
	parser.add_argument('--op', default=0, type=int)
	parser.add_argument('--map_func', default=0, type=int)
	parser.add_argument('--fast', default=1, type=int)
	parser.add_argument('--dump-timer', default=5, type=int)

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	return parser.parse_args()

if __name__ == '__main__':
	args = parse_args()
	args.test_dir = os.path.abspath('./datasets/FB15K237/per_type_temp/')
	config_file = os.path.abspath('./datasets/FB15K237/fb15k237.config')
	type_offset_file = os.path.abspath('./datasets/FB15K237/fb15k237_ko_0.07_offset.p')
	config = utils.read_config(config_file)
	t.cuda.set_device(0)
	
	type_offset = cPickle.load(open(type_offset_file))
	if args.model == 'HEER':
		model = HEER(type_offset_file, config_file)
	elif args.model == 'DistMult':
		model = DistMult('embeddings/distmult/fb15k237_DistMult.pt')
	else:
		model = HEER(type_offset_file, config_file)

	suffix = '_fb15k237_ko_0.07_eval.txt'

	in_mapping = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_0.07_in_mapping.p'))
	for idx, i in enumerate(config['types']):
		edge_prefix = []
		edge_prefix += (i, i+'-1')
		#print("Edge Type:", idx)
		#print(edge_prefix)

		tp = idx
		try:
			for prefix in edge_prefix:
				input_eval_file = args.test_dir + '/' + prefix + suffix
				with open(input_eval_file, 'r') as INPUT:
					_input = []
					_output = []
					for line in INPUT:
						node = line.strip().split(' ')
						_type_a, _id_a = node[0].split(':')
						_type_b, _id_b = node[1].split(':')
						#print(_type_a, _id_a)
						#if _id_a in in_mapping[_type_a] and _id_b in in_mapping[_type_b]:
						if config['edges'][idx][2] == 1 and '-1' in prefix:
							_output.append(in_mapping[_type_a][_id_a] + type_offset[_type_a])
							_input.append(in_mapping[_type_b][_id_b] + type_offset[_type_b])
						else:
							_input.append(in_mapping[_type_a][_id_a] + type_offset[_type_a])
							_output.append(in_mapping[_type_b][_id_b] + type_offset[_type_b])
						#else:
							#print(line)
						#	continue
					#if num_unseen_cases > 0:
					#	print("WARNING: " + str(num_unseen_cases) + " unseen cases exist in in_mapping, which are predicted to be 0.5.")


				if len(_input) == 0:
					print("no this type! in test")
					continue
				input_data = tdata.TensorDataset(t.LongTensor(_input), t.LongTensor(_output))
				#print(len(input_data))
				
				data_reader = tdata.DataLoader(input_data, args.batch_size, shuffle=False)
				score = []

				for i, data in enumerate(data_reader, 0):
					inputs, labels = data
					loss = model.predict(inputs, labels, tp)
					if isinstance(loss, float):
						loss = [loss] 
					score += loss

				with open(input_eval_file, 'r') as INPUT, open(input_eval_file.replace('_eval', '_pred'), 'w') as OUTPUT:
					num_unseen_cases = 0
					for i, line in enumerate(INPUT):
						node = line.strip().split(' ')
						_type_a, _id_a = node[0].split(':')
						_type_b, _id_b = node[1].split(':')
						assert _id_a in in_mapping[_type_a] and _id_b in in_mapping[_type_b]
						node[2] = str(score[i])
						OUTPUT.write(' '.join(node) + '\n')
		except IOError:
			print("No test case with edge type %s" % i)

