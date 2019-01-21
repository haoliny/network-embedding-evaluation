import numpy as np
import torch
from joblib import load
import torch.nn as nn
from torch.autograd import Variable

class Node2Vec(nn.Module):
	def __init__(self):
		super(Node2Vec, self).__init__()
		#freq_dict = np.load("./embeddings/node2vec/freq_dict.npy").item()
		#node_dict = np.load("./embeddings/node2vec/node_dict.npy").item()
		#self.freq_dict_train = np.load("./embeddings/node2vec/freq_dict_train.npy").item()
		#self.node_embed = np.load("./embeddings/node2vec/node_embedding.npy").item()
		#self.logreg_coef = np.load("./embeddings/node2vec/legreg_coef_dict.npy").item()
		self.logreg_models = []
		for edge_id in range(237):
			self.logreg_models.append(load('./embeddings/node2vec/logreg/edge_' + str(edge_id) + '_logreg.joblib'))

		self._n_entities = 14541
		self._n_relations = 128
		self.node_embed = np.zeros((self._n_entities, self._n_relations))
		with open('./embeddings/node2vec/fb15k237.emb', 'r') as INPUT:
			for line in INPUT.readlines():
				line_split = line.strip().split()
				self.node_embed[int(line_split[0])] = [float(dim_val) for dim_val in line_split[1:]]
		#print(self.node_embed)
		self._embed = nn.Embedding(self._n_entities, self._n_relations)
		self._embed.weight.data.copy_(torch.from_numpy(self.node_embed))
		self.cuda()
		"""self.entity_idx_to_n2v_idx = {}
		self.relation_idx_to_n2v_idx = {}
		with open('./datasets/FB15K237/entity2id.txt', 'r') as INPUT:
			for line in INPUT.readlines():
				pair = line.strip().split('\t')
				if len(pair) == 2 and pair[0] in node_dict:
					self.entity_idx_to_n2v_idx[int(pair[1])] = node_dict[pair[0]]
		with open('./datasets/FB15K237/relation2id.txt', 'r') as INPUT:
			for line in INPUT.readlines():
				pair = line.strip().split('\t')
				if len(pair) == 2 and pair[0] in freq_dict:
					self.relation_idx_to_n2v_idx[int(pair[1])] = freq_dict[pair[0]]
		
		self.n2v_idx_to_entity_idx = dict((v,k) for k,v in self.entity_idx_to_n2v_idx.iteritems())
		self.n2v_idx_to_relation_idx = dict((v,k) for k,v in self.relation_idx_to_n2v_idx.iteritems())"""

	def score(self, head, rel, tail):
		#x = self.node_embed[rel][head] + self.node_embed[rel][tail]
		#c_x = self.logreg_coef[rel][1]
		#c_0 = self.logreg_coef[rel][0]
		#f_x = c_0 + np.sum(np.multiply(c_x, x))
		#return 1/float(1 + np.exp(-f_x))
		h_embed = self._embed(Variable(head.cuda()))
		t_embed = self._embed(Variable(tail.cuda()))
		hammard_prod = h_embed * t_embed
		#print(self.logreg_models[rel].predict_proba(hammard_prod.data.tolist())[:,1])
		return self.logreg_models[rel].predict_proba(hammard_prod.data.tolist())[:, 1]

	def predict(self, h, t, r, pred_tail = True):
		#prob = []
		#h = self.entity_idx_to_n2v_idx[h]
		#t = self.entity_idx_to_n2v_idx[t]
		#r = self.relation_idx_to_n2v_idx[r]
		#nodes1 = [int(i[0]) for i in self.freq_dict_train[r] if i[0] != h and i[0] != t]
		#nodes2 = [int(i[1]) for i in self.freq_dict_train[r] if i[1] != h and i[1] != t]
		#nodes = list(set(nodes1+nodes2))
		#if pred_tail:
		#	samples = [t] + nodes
		#else:
		#	samples = [h] + nodes
		#for i in range(14505):
		#	if pred_tail:
		#		prob += [self.score(h, r, i)]
		#	else:
		#		prob += [self.score(i, r, t)]
		#print(prob[:10])
		if pred_tail:
			predict_h = torch.LongTensor([h] * self._n_entities)
			predict_t = torch.LongTensor(range(self._n_entities))
		else:
			predict_h = torch.LongTensor(range(self._n_entities))
			predict_t = torch.LongTensor([t] * self._n_entities)
		prob = self.score(predict_h, r, predict_t)
		return torch.sort(torch.Tensor(prob), descending=True)[1].data.cpu().numpy().tolist()

	def predict_relation(self, h, t, r):
		#h = self.entity_idx_to_n2v_idx[h]
		#t = self.entity_idx_to_n2v_idx[t]
		#r = self.relation_idx_to_n2v_idx[r]

		predict_h = torch.LongTensor([h])
		predict_t = torch.LongTensor([t])
		#relations, prob = [], []
		prob = np.empty(237)
		for r in range(237):
			prob[r] = self.score(predict_h, r, predict_t)[0]
		return np.argsort(prob)[::-1]
			#if h in self.node_embed[i] and t in self.node_embed[i]:
			#prob += [self.score(h, i, t)]
			#relations += [i]
		#if len(relations) == 0:
			#raise Exception
		#return [i for i in sorted(range(len(prob)), key=lambda k: prob[k], reverse=True)]