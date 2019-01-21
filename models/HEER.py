import torch as t
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np
import cPickle
import os
import util.utils as utils

class HEER(nn.Module):
    def __init__(self, type_offset, config):
        super(HEER, self).__init__()
        t.cuda.set_device(0)

        embedding_file = os.path.abspath('embeddings/heer/heer_fb15k237_ko_0.07_60_op_1_mode_0_rescale_0.1_lr_10_lrr_10.pt')
        #config = utils.read_config(config_file)
        #type_offset = cPickle.load(open(type_offset_file))

        self.num_classes = type_offset['sum']
        self.type_offset = []
        self.mode = 1
        self.map_mode = 0
        for tp in config['nodes']:
            if tp in type_offset:
                self.type_offset.append(type_offset[tp])

        self.edge_types = config['edges']
        self.embed_size = 128
        self.in_embed = nn.Embedding(self.num_classes, self.embed_size, sparse=True)

        self.edge_mapping = nn.ModuleList()
        self.edge_mapping_bn = nn.ModuleList()
        self.out_embed = nn.Embedding(self.num_classes, self.embed_size, sparse=True)

        self.out_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-0.1, 0.1).cuda())
        self.in_embed.weight = Parameter(t.FloatTensor(self.num_classes, self.embed_size).uniform_(-0.1, 0.1).cuda())

        if self.map_mode > -1: 
            for tp in self.edge_types:
                self.edge_mapping.append(self.genMappingLayer(self.map_mode))
                """
                if self.map_mode > 0:
                    self.edge_mapping_bn.append(nn.BatchNorm1d(self.embed_size, affine=True).cuda())
                    self.edge_mapping_bn[-1].weight.data.fill_(1)
                    self.edge_mapping_bn[-1].bias.data.zero_()
                    self.edge_mapping_bn[-1].register_parameter('bias', None)
                """

                #if self.mode == -2:
                    #self.edge_mapping_bn.append(nn.Dropout().cuda())
        xxx = t.load(embedding_file, map_location=lambda storage, loc: storage)
        self.load_state_dict(xxx)
        self.cuda()
        self.eval()
        
        self.type_offset.append(type_offset['sum'])

    def genMappingLayer(self, mode):
        """
        mode -4: vanilla linear on addition
        mode -3: vanilla linear on deduction
        mode -2: vanilla linear on outer-product
        mode -1: unimetric
        mode 0: vanilla linear(scale) layer
        mode 1: vanilla batch normalization layer
        mode 2: deeper metric
        """
        _layer = None
        if mode == -1:
            return _layer
        else:
            if mode == 0:
                _layer = utils.DiagLinear(self.embed_size).cuda()
                _layer.weight = Parameter(t.FloatTensor(self.embed_size).fill_(1.0).cuda())
            if mode == 2:
                _layer = utils.DeepSemantics(self.embed_size, 20, 50, bias=False).cuda()
        return _layer

    def edge_map(self, x, tp):
        if self.map_mode == -1:
            return x
        else:
            return self.edge_mapping[tp](x)

    def edge_rep(self, input_a, input_b):
        #mode 1: hadamard-product
        #mode 2: outer-product
        #mode 3: deduction
        #mode 4: addition
        if self.mode == 1:
            return input_a * input_b
        elif self.mode == 2:
            return t.bmm(input_a.unsqueeze(2), input_b.unsqueeze(1)).view(-1, self.embed_size ** 2) + t.bmm(input_b.unsqueeze(2), input_a.unsqueeze(1)).view(-1, self.embed_size ** 2)
        elif self.mode == 3:
            return (input_a - input_b) ** 2
        elif self.mode == 4:
            return (input_a + input_b) ** 2
        else:
            return input_a * input_b

    """
    Score triplets that have the same edge type `tp` but varying head/tail nodes
    Args:
        inputs - torch.LongTensor array of head nodes
        outputs - torch.LongTensor array of tail nodes
        tp - single int of the edge type
    Return:
        a list of scores corresponding to each <input,output,tp> triplet
    """
    def predict(self, head, tail, tp, pred_tail=True):
        if pred_tail:
            inputs = t.LongTensor([head] * self.num_classes)
            outputs = t.LongTensor(range(self.num_classes))
        else:
            inputs = t.LongTensor(range(self.num_classes))
            outputs = t.LongTensor([tail] * self.num_classes)
        use_cuda = True
        if use_cuda:
            inputs = inputs.cuda()
            outputs = outputs.cuda()

        u_input = self.in_embed(Variable(inputs))
        v_output = self.out_embed(Variable(outputs))
        log_target = 0.0
        if self.edge_types[tp][2] == 0:
            
            u_output = self.out_embed(Variable(inputs))
            v_input = self.in_embed(Variable(outputs))

            log_target = self.edge_map(self.edge_rep(u_input, v_input), tp).sum(1).squeeze().sigmoid() + self.edge_map(self.edge_rep(u_output, v_output), tp).sum(1).squeeze().sigmoid()
            log_target /= 2
        else:
            log_target = self.edge_map(self.edge_rep(u_input, v_output), tp).sum(1).squeeze().sigmoid()
        #log_target = (input * output).sum(1).squeeze().sigmoid()
        
        return t.sort(log_target, descending=True)[1].data.cpu().numpy().tolist()

    def predict_relation(self, head, tail, tp):
        heads = t.LongTensor([head])
        tails = t.LongTensor([tail])
        tps = range(len(self.edge_types))
        inputs = heads.cuda()
        outputs = tails.cuda()

        u_input = self.in_embed(Variable(inputs))
        v_output = self.out_embed(Variable(outputs))
        score = []
        for tp in tps:
            log_target = 0.0
            if self.edge_types[tp][2] == 0:
                u_output = self.out_embed(Variable(inputs))
                v_input = self.in_embed(Variable(outputs))

                log_target = self.edge_map(self.edge_rep(u_input, v_input), tp).sum(1).squeeze().sigmoid() + self.edge_map(self.edge_rep(u_output, v_output), tp).sum(1).squeeze().sigmoid()
                log_target /= 2
            else:
                log_target = self.edge_map(self.edge_rep(u_input, v_output), tp).sum(1).squeeze().sigmoid()
            score.append(log_target.data.cpu().numpy().tolist())
        return np.argsort(np.array(score)).tolist()[::-1]


    def input_embeddings(self):
        return self.in_embed.weight.data.cpu().numpy()
    def output_embeddings(self):
        return self.out_embed.weight.data.cpu().numpy()

    