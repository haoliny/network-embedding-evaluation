import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable


class TransE(nn.Module):
	def __init__(self, embedding_file):
		super(TransE,self).__init__()
		self._n_entities = 14505
		self._n_relations = 237
		self._embed_dim = 128
		self.ent_embeddings=nn.Embedding(self._n_entities,self._embed_dim)
		self.rel_embeddings=nn.Embedding(self._n_relations,self._embed_dim)
		self.load_state_dict(torch.load(embedding_file))
		print('loaded saved TransE embedding', embedding_file)
		self.cuda()
	
	r'''
	TransE is the first model to introduce translation-based embedding, 
	which interprets relations as the translations operating on entities.
	'''
	def _calc(self,h,t,r):
		return torch.abs(h + r - t)

	def predict(self, predict_h, predict_t, predict_r, pred_tail=True):
		if pred_tail:
			predict_h = torch.LongTensor([predict_h] * self._n_entities)
			predict_t = torch.LongTensor(range(self._n_entities))
		else:
			predict_h = torch.LongTensor(range(self._n_entities))
			predict_t = torch.LongTensor([predict_t] * self._n_entities)
		predict_r = torch.LongTensor([predict_r] * self._n_entities)

		p_h=self.ent_embeddings(Variable(predict_h).cuda())
		p_t=self.ent_embeddings(Variable(predict_t).cuda())
		p_r=self.rel_embeddings(Variable(predict_r).cuda())
		_p_score = -self._calc(p_h, p_t, p_r)
		p_score=torch.sum(_p_score,1)
		return torch.sort(p_score, descending=True)[1].data.cpu().numpy().tolist()

	def predict_relation(self, head, tail, rs):
		heads = torch.LongTensor([head] * self._n_relations).cuda()
		tails = torch.LongTensor([tail] * self._n_relations).cuda()
		rs = torch.LongTensor(range(self._n_relations)).cuda()

		p_e_h = self.ent_embeddings(Variable(heads.cuda()))
		p_e_t = self.ent_embeddings(Variable(tails.cuda()))
		p_e_r = self.rel_embeddings(Variable(rs.cuda()))
		_p_score = -self._calc(p_e_h,p_e_t,p_e_r)
		p_score=torch.sum(_p_score,1)
		return torch.sort(p_score, descending=True)[1].data.cpu().numpy().tolist()