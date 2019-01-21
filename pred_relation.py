import numpy as np
import cPickle
import sys,os
import argparse
import ast
from models.HEER import HEER
from models.DistMult import DistMult
from models.ComplEx import ComplEx
from models.Node2Vec import Node2Vec
from models.TransE import TransE
from models.DeepWalk import DeepWalk


global type_offset
global in_mapping
global train_edge_dict

def read_config(conf_name):
	config = {}
	with open(conf_name) as IN:
		config['edges'] = ast.literal_eval(IN.readline())
		config['nodes'] = ast.literal_eval(IN.readline())
		config['types'] = ast.literal_eval(IN.readline())
		for i,x in enumerate(ast.literal_eval(IN.readline())):
			config['edges'][i].append(int(x))        
	assert len(config['edges']) == len(config['types'])
	return config

def parse_args():
	'''
	Parses the relation prediction arguments.
	'''
	parser = argparse.ArgumentParser(description="Run FB15K-237 relation prediction tests on a specified model.")
	parser.add_argument('--model', nargs='?', required=True,
						help='Which specific model to run')
	parser.add_argument('--output-file', nargs='?', required=True,
						help='Metrics output file')

	return parser.parse_args()

def calculate_metrics(scores, head_type, head_val, tail_type, tail_val, edge_type_str, edge_type_idx, edge_id_str_map=None):
	num_grtr, num_grtr_f = -1, 0
	avg_precision = 0.0
	true_pos_cnt, false_pos_cnt = 0.0, 0.0
	for pred_rank, pred_id in enumerate(scores):
		if pred_id == edge_type_idx:
			num_grtr = pred_rank
		#compute average precision
		pred_str = str(pred_id) + ':d'
		if edge_id_str_map:
			pred_str = edge_id_str_map[pred_id]
		if pred_str == edge_type_str or pred_str in train_edge_dict[head_type][head_val]:
			true_pos_cnt += 1
			avg_precision += true_pos_cnt/(true_pos_cnt + false_pos_cnt)
		else:
			false_pos_cnt += 1
			num_grtr_f += (num_grtr == -1)

	rr = 1.0/(1+num_grtr)
	rr_f = 1.0/(1+num_grtr_f)
	#print('target score rank = ', 1.0/(num_grtr+1), rr, num_less)
	hit1 = float(num_grtr<1)
	hit3 = float(num_grtr<3)
	hit10 = float(num_grtr<10)
	hit1_f = float(num_grtr_f<1)
	hit3_f = float(num_grtr_f<3)
	hit10_f = float(num_grtr_f<10)
	return rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f, avg_precision/true_pos_cnt

if __name__ == '__main__':
	args = parse_args()
	config_file = os.path.abspath('./datasets/FB15K237/fb15k237.config')
	type_offset_file = os.path.abspath('./datasets/FB15K237/fb15k237_ko_0.07_offset.p')
	config = read_config(config_file)

	type_offset = cPickle.load(open(type_offset_file))

	if args.model == 'HEER':
		print('Loading HEER model')
		model = HEER(type_offset, config)
	elif args.model == 'DistMult':
		print('Loading DistMult model')
		model = DistMult('embeddings/distmult/distmult.pt')
	elif args.model == 'ComplEx':
		print('Loading ComplEx model')
		model = ComplEx('embeddings/complex/fb15k237_ComplEx.pt')
	elif args.model == 'Node2Vec':
		print('Loading Node2Vec model')
		model = Node2Vec()
	elif args.model == 'DeepWalk':
		print('Loading DeepWalk model')
		model = DeepWalk()
	elif args.model == 'TransE':
		print('Loading TransE model')
		model = TransE('embeddings/transe/transe.pt')
	else:
		model = HEER(type_offset, config)
	suffix = '_fb15k237_ko_0.07_eval.txt'

	in_mapping = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_0.07_in_mapping.p'))
	train_edge_dict = cPickle.load(open('./datasets/FB15K237/fb15k237_train_edge_dict.p'))
	node_type_dict = cPickle.load(open('./datasets/FB15K237/fb15k237_node_type_dict.p'))
	ko_dic = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_dict.p'))

	metrics = {
		'predict_relation': {
			'rr_sum': 0.0,
			'hit1_sum': 0.0,
			'hit3_sum': 0.0,
			'hit10_sum': 0.0,
			'rr_f_sum': 0.0,
			'hit1_f_sum': 0.0,
			'hit3_f_sum': 0.0,
			'hit10_f_sum': 0.0,
			'avg_precision': 0.0
		},
	}
	test_count = 0
	with open('./datasets/FB15K237/fb15k237_test.hin') as INPUT:
		for line in INPUT.readlines()[:50]:
			node = line.strip().split(' ')
			print('processing test case %d' % test_count)
			_type_a, _id_a = node[0].split(':')
			_type_b, _id_b = node[1].split(':')
			edge_id_str_arr = None # non-HEER models map edge IDs directly to their indices
			if args.model == 'HEER':
				# HEER model needs ID of edge as stored in its config
				edge_type = config['types'].index(node[3])
				edge_id_str_arr = config['types']
			else:
				# other models use ID of edge taken directly from the file
				edge_type = int(node[3].split(':')[0])
			if _id_a not in train_edge_dict[_type_a] or _id_b not in train_edge_dict[_type_b]:
				print('test_head_value %s or test_tail_value %s not valid' % (_id_a, _id_b))
				continue

			head = in_mapping[_type_a][_id_a] + type_offset[_type_a]
			tail = in_mapping[_type_b][_id_b] + type_offset[_type_b]
			try:
				# Test relation prediction
				scores = model.predict_relation(head, tail, edge_type)
				rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f, avg_prec = calculate_metrics(scores, _type_a, _id_a, _type_b, _id_b, node[3], edge_type, edge_id_str_arr)
				metrics['predict_relation']['rr_sum'] += rr
				metrics['predict_relation']['hit1_sum'] += hit1
				metrics['predict_relation']['hit3_sum'] += hit3
				metrics['predict_relation']['hit10_sum'] += hit10
				metrics['predict_relation']['rr_f_sum'] += rr_f
				metrics['predict_relation']['hit1_f_sum'] += hit1_f
				metrics['predict_relation']['hit3_f_sum'] += hit3_f
				metrics['predict_relation']['hit10_f_sum'] += hit10_f
				metrics['predict_relation']['avg_precision'] += avg_prec

				test_count += 1
			except:
				print('Model was unable to test the triple <%d,%d,%d>' % (head, edge_type, tail))
		#print('Finished scoring edge %s, rr_sum = %f, test_count = %d' % (node[3], rr_sum, test_count))

	with open(args.output_file, 'w') as OUTPUT:
		OUTPUT.write('total tests = %d\n' % test_count)
		OUTPUT.write('metric:\t\t\tMRR\t\thit@1\t\thit@3\t\thit@10\n')
		OUTPUT.write('relation (raw)\t\t%f\t%f\t%f\t%f\n' % (
			metrics['predict_relation']['rr_sum']/test_count,
			metrics['predict_relation']['hit1_sum']/test_count,
			metrics['predict_relation']['hit3_sum']/test_count,
			metrics['predict_relation']['hit10_sum']/test_count))
		OUTPUT.write('relation (filtered)\t%f\t%f\t%f\t%f\n' % (
			metrics['predict_relation']['rr_f_sum']/test_count,
			metrics['predict_relation']['hit1_f_sum']/test_count,
			metrics['predict_relation']['hit3_f_sum']/test_count,
			metrics['predict_relation']['hit10_f_sum']/test_count))
		OUTPUT.write('\n')
		OUTPUT.write('relation mAP = %f\n' % (metrics['predict_relation']['avg_precision']/test_count))
	print('Results written to %s' % args.output_file)