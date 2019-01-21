import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class DistMult(nn.Module):
	def __init__(self, embedding_file):
		torch.cuda.set_device(0)
		super(DistMult, self).__init__()
		self._n_entities = 14541
		self._n_relations = 237
		self.ent_embeddings=nn.Embedding(self._n_entities, 128)
		self.rel_embeddings=nn.Embedding(self._n_relations, 128)
		self.load_state_dict(torch.load(embedding_file))
		print('loaded saved DistMult embedding', embedding_file)
		self.cuda()
		#self.eval()
	
	# score function of DistMult
	def _calc(self, h, t, r):
		return torch.sum(h*t*r,1,False)

	# either predict_h is an array of the same node and predict_t is an array of
	# all nodes, or vice versa, for a given test case
	def predict(self, predict_h, predict_t, predict_r, pred_tail=True):
		if pred_tail:
			predict_h = torch.LongTensor([predict_h] * self._n_entities)
			predict_t = torch.LongTensor(range(self._n_entities))
		else:
			predict_h = torch.LongTensor(range(self._n_entities))
			predict_t = torch.LongTensor([predict_t] * self._n_entities)
		predict_r = torch.LongTensor([predict_r] * self._n_entities)

		p_e_h = self.ent_embeddings(Variable(predict_h.cuda()))
		p_e_t = self.ent_embeddings(Variable(predict_t.cuda()))
		p_e_r = self.rel_embeddings(Variable(predict_r.cuda()))
		p_score = self._calc(p_e_h,p_e_t,p_e_r)
		return torch.sort(p_score, descending=True)[1].data.cpu().numpy().tolist()

	def predict_relation(self, head, tail, rs):
		heads = torch.LongTensor([head] * self._n_relations).cuda()
		tails = torch.LongTensor([tail] * self._n_relations).cuda()
		rs = torch.LongTensor(range(self._n_relations)).cuda()

		p_e_h = self.ent_embeddings(Variable(heads.cuda()))
		p_e_t = self.ent_embeddings(Variable(tails.cuda()))
		p_e_r = self.rel_embeddings(Variable(rs.cuda()))
		p_score = self._calc(p_e_h,p_e_t,p_e_r)
		return torch.sort(p_score, descending=True)[1].data.cpu().numpy().tolist()