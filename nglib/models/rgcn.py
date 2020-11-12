import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import utils as dgl_utils

logger = logging.getLogger(__name__)


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, use_bias=True, activation=None, 
                 self_loop=False, dropout=0.2, use_gate=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat 
        self.out_feat = out_feat
        self.num_rels = num_rels 
        self.use_bias = use_bias
        self.activation = activation
        self.use_self_loop = self_loop 
        self.dropout = dropout
        self.use_gate = use_gate

        # relation weights
        self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat. self.out_feat))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

        # message func
        # self.message_func = self.basis_message_func

        # bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.bias)

        # weight for self loop
        if self.use_self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            logger.info('loop_weight: {}'.format(self.loop_weight.shape))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

        if self.use_gate:
            self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels, self.in_feat, 1))
            nn.init.xavier_uniform_(self.gate_weight, gain=nn.init.calculate_gain('sigmoid'))

    def forward(self, g):
        assert g.is_homograph(), "not a homograph; convert it with to_homo and pass in the edge type as argument"

        with g.local_scope():
            if self.use_self_loop:
                loop_message = dgl_utils.bmm_maybe_select(g.ndata['h'], self.loop_weight)

            # message passing
            # g.update_all(self.message_func, fn.sum(msg='msg', out='h'))

            # apply bias and activation
            node_repr = g.ndata['h']
            if self.use_bias:
                node_repr += self.bias
            if self.use_self_loop:
                node_repr += loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self.dropout(node_repr)
            return node_repr


class RGCNModel(nn.Module):
    def __init__(self, in_dim, h_dim, num_nodes, num_rels, num_hidden_layers=2, 
                 dropout=0.2, use_self_loop=True, use_gate=True):
        super(RGCNModel, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_gate = use_gate

        # create rgcn layer
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        for idx in range(self.num_hidden_layers)
        

        
        
