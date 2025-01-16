import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn
import dgl.function as fn
import numpy as np


def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func

def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

# To copy edge features to be passed to FFN_e
def out_edge_features(edge_feat):
    def func(edges):
        return {'e_out': edges.data[edge_feat]}
    return func


def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {field: torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        if use_bias:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=True)
        else:
            self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
            self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=False)
    
    def propagate_attention(self, g):
        # Compute attention score
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score')) #, edges)
        # scaling
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        # Use available edge features to modify the scores
        g.apply_edges(imp_exp_attn('score', 'proj_e'))
        # Copy edge features as e_out to be passed to FFN_e
        g.apply_edges(out_edge_features('score'))
        # softmax
        g.apply_edges(exp('score'))
        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score', 'score'), fn.sum('score', 'z'))
    
    def forward(self, g, h, e):
        Q_h = self.Q(h)
        K_h = self.K(h)
        V_h = self.V(h)
        proj_e = self.proj_e(e)
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)
        g.edata['proj_e'] = proj_e.view(-1, self.num_heads, self.out_dim)
        self.propagate_attention(g)
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6)) # adding eps to all values here
        e_out = g.edata['e_out']
        return h_out, e_out
    
    
class WeightedGATConv(dglnn.GATConv):
    def __init__(self, in_feats, out_feats, num_heads, bias=True):
        super(WeightedGATConv, self).__init__(in_feats, out_feats, num_heads, bias)
        
    def forward(self, graph, feat, edge_weight):
        with graph.local_scope():
            graph.edata['weight'] = edge_weight
            out = super(WeightedGATConv, self).forward(graph, feat)
            out = out.view(out.shape[0], 8, -1)
            out = out.mean(dim=1)  
            return out
        
        
class GraphTransformerLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, dropout, layer_norm=True, batch_norm=False, residual=True, use_bias=False):
        super().__init__()
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm   
        self.attention = MultiHeadAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias)
        self.O_h = nn.Linear(out_dim, out_dim)
        self.O_e = nn.Linear(out_dim, out_dim)
        self.layer_norm1_h = nn.LayerNorm(out_dim)
        self.layer_norm1_e = nn.LayerNorm(out_dim)
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)
        # FFN for e
        self.FFN_e_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_e_layer2 = nn.Linear(out_dim*2, out_dim)
        self.layer_norm2_h = nn.LayerNorm(out_dim)
        self.layer_norm2_e = nn.LayerNorm(out_dim)      
        self.Residual_GCN_fc = nn.Sequential(
            nn.Linear(out_dim, out_dim, bias=False),
            nn.LayerNorm(out_dim),
            nn.ReLU(inplace=True),
        )
        self.Residual_GCN_layer = dglnn.GraphConv(out_dim, out_dim, bias=False)
        
        
    def forward(self, g, h, e):
        h_in1 = h # for first residual connection
        e_in1 = e # for first residual connection
        # multi-head attention out
        h_attn_out, e_attn_out = self.attention(g, h, e)
        h = h_attn_out.view(-1, self.out_channels)
        e = e_attn_out.view(-1, self.out_channels)
        h = F.dropout(h, self.dropout, training=self.training)
        e = F.dropout(e, self.dropout, training=self.training)
        h = self.O_h(h)
        e = self.O_e(e)
        h = h_in1 + h # residual connection
        e = e_in1 + e # residual connection
        
        x1 = self.Residual_GCN_layer(g, h, edge_weight=e)
        x1 = F.relu(x1, inplace=True)
        f1 = self.Residual_GCN_fc(h)
        h = x1 + f1

        h = self.layer_norm1_h(h)
        h_in2 = h # for second residual connection
        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)
        h = h_in2 + h # residual connection    
        h = self.layer_norm2_h(h)    
        return h
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)