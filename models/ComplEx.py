import torch
import torch.autograd as autograd
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

class ComplEx(nn.Module):
	def __init__(self,embedding_file):
		torch.cuda.set_device(0)
		super(ComplEx,self).__init__()
		self._n_entities = 14541
		self._n_relations = 237
		self.ent_re_embeddings=nn.Embedding(self._n_entities,128)
		self.ent_im_embeddings=nn.Embedding(self._n_entities,128)
		self.rel_re_embeddings=nn.Embedding(self._n_relations,128)
		self.rel_im_embeddings=nn.Embedding(self._n_relations,128)
		self.load_state_dict(torch.load(embedding_file))
		self.cuda()
		self.eval()

	#score function of ComplEx
	def _calc(self,e_re_h,e_im_h,e_re_t,e_im_t,r_re,r_im):
		return torch.sum(r_re * e_re_h * e_re_t + r_re * e_im_h * e_im_t + r_im * e_re_h * e_im_t - r_im * e_im_h * e_re_t,1,False)

	def predict(self, predict_h, predict_t, predict_r, pred_tail=True):
		if pred_tail:
			predict_h = torch.LongTensor([predict_h] * self._n_entities)
			predict_t = torch.LongTensor(range(self._n_entities))
		else:
			predict_h = torch.LongTensor(range(self._n_entities))
			predict_t = torch.LongTensor([predict_t] * self._n_entities)
		predict_r = torch.LongTensor([predict_r] * self._n_entities)

		p_re_h=self.ent_re_embeddings(Variable(predict_h.cuda()))
		p_re_t=self.ent_re_embeddings(Variable(predict_t.cuda()))
		p_re_r=self.rel_re_embeddings(Variable(predict_r.cuda()))
		p_im_h=self.ent_im_embeddings(Variable(predict_h.cuda()))
		p_im_t=self.ent_im_embeddings(Variable(predict_t.cuda()))
		p_im_r=self.rel_im_embeddings(Variable(predict_r.cuda()))
		p_score = self._calc(p_re_h, p_im_h, p_re_t, p_im_t, p_re_r, p_im_r)
		return torch.sort(p_score, descending=True)[1].data.tolist()

	def predict_relation(self, head, tail, edge):
		predict_h = torch.LongTensor([head] * self._n_relations)
		predict_t = torch.LongTensor([tail] * self._n_relations)
		predict_r = torch.LongTensor(range(self._n_relations))

		p_re_h=self.ent_re_embeddings(Variable(predict_h.cuda()))
		p_re_t=self.ent_re_embeddings(Variable(predict_t.cuda()))
		p_re_r=self.rel_re_embeddings(Variable(predict_r.cuda()))
		p_im_h=self.ent_im_embeddings(Variable(predict_h.cuda()))
		p_im_t=self.ent_im_embeddings(Variable(predict_t.cuda()))
		p_im_r=self.rel_im_embeddings(Variable(predict_r.cuda()))
		p_score = self._calc(p_re_h, p_im_h, p_re_t, p_im_t, p_re_r, p_im_r)
		return torch.sort(p_score, descending=True)[1].data.tolist()
