import numpy as np
import cPickle
import sys,os
import argparse
import ast
from models.HEER import HEER
from models.DistMult import DistMult
from models.ComplEx import ComplEx
from models.Node2Vec import Node2Vec
from models.DeepWalk import DeepWalk
from models.TransE import TransE

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
	Parses the entity prediction arguments.
	'''
	parser = argparse.ArgumentParser(description="Run FB15K-237 entity prediction tests on a specified model.")
	parser.add_argument('--model', nargs='?', required=True,
						help='Which specific model to run')
	parser.add_argument('--output-file', nargs='?', required=True,
						help='Metrics output file')

	return parser.parse_args()

def calculate_metrics(batch, head_type, head_val, tail_type, tail_val, edge_type, pred_tail=True):
	if not pred_tail:
		target_idx = in_mapping[head_type][head_val] + type_offset[head_type]
	else:
		target_idx = in_mapping[tail_type][tail_val] + type_offset[tail_type]

	num_grtr, num_grtr_f = -1, 0
	avg_precision = 0.0
	true_pos_cnt, false_pos_cnt = 0.0, 0.0

	for pred_rank, pred_id in enumerate(batch):
		if pred_id == target_idx:
			num_grtr = pred_rank
		#compute average precision
		if pred_tail:
			if str(pred_id) == tail_val or (
				str(pred_id) in train_edge_dict[head_type][head_val] and
				edge_type in train_edge_dict[head_type][head_val][str(pred_id)]):
				true_pos_cnt += 1
				avg_precision += true_pos_cnt/(true_pos_cnt + false_pos_cnt)
				#print(str(pred_id) + ' is a correct prediction')
			else:
				false_pos_cnt += 1
				num_grtr_f += (num_grtr == -1)
		else:
			if str(pred_id) == head_val or (
				str(pred_id) in train_edge_dict[head_type] and
				tail_val in train_edge_dict[head_type][str(pred_id)] and
				edge_type in train_edge_dict[head_type][str(pred_id)][tail_val]):
				true_pos_cnt += 1
				avg_precision += true_pos_cnt/(true_pos_cnt + false_pos_cnt)
			else:
				false_pos_cnt += 1
				num_grtr_f += (num_grtr == -1)

	rr = 1.0/(1+num_grtr)
	rr_f = 1.0/(1+num_grtr_f)
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
		print('Loading HEER model')
		model = HEER(type_offset, config)


	in_mapping = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_0.07_in_mapping.p'))
	train_edge_dict = cPickle.load(open('./datasets/FB15K237/fb15k237_train_edge_dict.p'))
	node_type_dict = cPickle.load(open('./datasets/FB15K237/fb15k237_node_type_dict.p'))
	ko_dic = cPickle.load(open('./datasets/FB15K237/fb15k237_ko_dict.p'))

	metrics = {
		'predict_head': {
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
		'predict_tail': {
			'rr_sum': 0.0,
			'hit1_sum': 0.0,
			'hit3_sum': 0.0,
			'hit10_sum': 0.0,
			'rr_f_sum': 0.0,
			'hit1_f_sum': 0.0,
			'hit3_f_sum': 0.0,
			'hit10_f_sum': 0.0,
			'avg_precision': 0.0
		}
	}
	test_count = 0

	with open('./datasets/FB15K237/fb15k237_test.hin') as INPUT:
		for line in INPUT.readlines()[:50]:
			node = line.strip().split(' ')
			print('processing test case %d' % test_count)
			_type_a, _id_a = node[0].split(':')
			_type_b, _id_b = node[1].split(':')
			if args.model == 'HEER':
				# HEER model needs ID of edge as stored in its config
				edge_type = config['types'].index(node[3])
			else:
				# other models use ID of edge taken directly from the file
				edge_type = int(node[3].split(':')[0])
			if _id_a not in train_edge_dict[_type_a] or _id_b not in train_edge_dict[_type_b]:
				print('test_head_value %s or test_tail_value %s not valid' % (_id_a, _id_b))
				continue

			head = in_mapping[_type_a][_id_a] + type_offset[_type_a]
			tail = in_mapping[_type_b][_id_b] + type_offset[_type_b]
			try:
				# Test head prediction
				scores = model.predict(head, tail, edge_type, False)
				rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f, avg_prec = calculate_metrics(scores, _type_a, _id_a, _type_b, _id_b, node[3], False)
				metrics['predict_head']['rr_sum'] += rr
				metrics['predict_head']['hit1_sum'] += hit1
				metrics['predict_head']['hit3_sum'] += hit3
				metrics['predict_head']['hit10_sum'] += hit10
				metrics['predict_head']['rr_f_sum'] += rr_f
				metrics['predict_head']['hit1_f_sum'] += hit1_f
				metrics['predict_head']['hit3_f_sum'] += hit3_f
				metrics['predict_head']['hit10_f_sum'] += hit10_f
				metrics['predict_head']['avg_precision'] += avg_prec

				# Test tail prediction
				scores = model.predict(head, tail, edge_type, True)
				rr, hit1, hit3, hit10, rr_f, hit1_f, hit3_f, hit10_f, avg_prec = calculate_metrics(scores, _type_a, _id_a, _type_b, _id_b, node[3], True)
				metrics['predict_tail']['rr_sum'] += rr
				metrics['predict_tail']['hit1_sum'] += hit1
				metrics['predict_tail']['hit3_sum'] += hit3
				metrics['predict_tail']['hit10_sum'] += hit10
				metrics['predict_tail']['rr_f_sum'] += rr_f
				metrics['predict_tail']['hit1_f_sum'] += hit1_f
				metrics['predict_tail']['hit3_f_sum'] += hit3_f
				metrics['predict_tail']['hit10_f_sum'] += hit10_f
				metrics['predict_tail']['avg_precision'] += avg_prec
				test_count += 1
			except:
				print('Model was unable to test the triple <%d,%d,%d>' % (head, edge_type, tail))

	with open(args.output_file, 'w') as OUTPUT:
		OUTPUT.write('total tests = %d\n' % test_count)
		OUTPUT.write('metric:\t\t\tMRR\t\thit@1\t\thit@3\t\thit@10\n')
		OUTPUT.write('head (raw)\t\t%f\t%f\t%f\t%f\n' % (
			metrics['predict_head']['rr_sum']/test_count,
			metrics['predict_head']['hit1_sum']/test_count,
			metrics['predict_head']['hit3_sum']/test_count,
			metrics['predict_head']['hit10_sum']/test_count))
		OUTPUT.write('tail (raw)\t\t%f\t%f\t%f\t%f\n' % (
			metrics['predict_tail']['rr_sum']/test_count,
			metrics['predict_tail']['hit1_sum']/test_count,
			metrics['predict_tail']['hit3_sum']/test_count,
			metrics['predict_tail']['hit10_sum']/test_count))
		OUTPUT.write('head (filtered)\t\t%f\t%f\t%f\t%f\n' % (
			metrics['predict_head']['rr_f_sum']/test_count,
			metrics['predict_head']['hit1_f_sum']/test_count,
			metrics['predict_head']['hit3_f_sum']/test_count,
			metrics['predict_head']['hit10_f_sum']/test_count))
		OUTPUT.write('tail (filtered)\t\t%f\t%f\t%f\t%f\n' % (
			metrics['predict_tail']['rr_f_sum']/test_count,
			metrics['predict_tail']['hit1_f_sum']/test_count,
			metrics['predict_tail']['hit3_f_sum']/test_count,
			metrics['predict_tail']['hit10_f_sum']/test_count))
		OUTPUT.write('\n')
		OUTPUT.write('head mAP = %f\n' % (metrics['predict_head']['avg_precision']/test_count))
		OUTPUT.write('tail mAP = %f\n' % (metrics['predict_tail']['avg_precision']/test_count))
	print('Results written to %s' % args.output_file)

