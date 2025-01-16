import torch.nn as nn
import torch
from models import gt_net_protein
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Embedding Size
d_model = 256
d_drop = 0.3

class TargetRepresentation_old(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num * 4, padding_idx=0)
        self.block_num = block_num
        # 定义 1D CNN 层
        self.cnn = nn.Sequential(
            nn.Conv1d(embedding_num * 4, embedding_num * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_num * 4, embedding_num * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(embedding_num * 4, embedding_num * 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x 的形状是 (batch_size, sequence_length)
        x = self.embed(x)  # 变为 (batch_size, sequence_length, embedding_num * 4)
        x = x.permute(0, 2, 1)  # 改变形状为 (batch_size, embedding_num * 4, sequence_length)
        x = self.cnn(x)  # 经过 1D CNN 层
        x = x.permute(0, 2, 1)  # 改回 (batch_size, sequence_length, embedding_num * 4)
        return x
    
    
class TargetRepresentation(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_num):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_num * 4, padding_idx=0)
        self.block_num = block_num
        # 定义 1D CNN 层
        self.conv1 = nn.Conv1d(embedding_num * 4, embedding_num * 4, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(embedding_num * 4, embedding_num * 4, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(embedding_num * 4, embedding_num * 4, kernel_size=7, padding=3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x 的形状是 (batch_size, sequence_length)
        x = self.embed(x)  # 变为 (batch_size, sequence_length, embedding_num * 4)
        x = x.permute(0, 2, 1)  # 改变形状为 (batch_size, embedding_num * 4, sequence_length)
        x1 = self.relu(self.conv1(x)) # 经过 1D CNN 层
        x2 = self.relu(self.conv2(x1)) # 经过 1D CNN 层
        x3 = self.relu(self.conv3(x2)) # 经过 1D CNN 层
        x = x1 + x2 + x3
        x = x.permute(0, 2, 1)  # 改回 (batch_size, sequence_length, embedding_num * 4)
        return x.squeeze(0)


class DTS_GraphTransformer(nn.Module):
    def __init__(self, block_num = 3, vocab_protein_size = 25 + 1, in_dim = 36):
        super(DTS_GraphTransformer,self).__init__()
        self.protein_dim = 128
        self.protein_encoder = TargetRepresentation(block_num, vocab_protein_size, 256)
        self.protein_gt = gt_net_protein.GraphTransformer(device, n_layers=1, node_dim=2084, edge_dim=6, hidden_dim=256,
                                                       out_dim=256, n_heads=8, in_feat_dropout=d_drop, dropout=d_drop, pos_enc_dim=8)
        self.dropout = nn.Dropout(0.3)
        self.linear1 = nn.Linear(d_model, 64)
        self.linear2 = nn.Linear(64, 16)
        self.linear3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Sigmoid()
        
    
    def forward(self, graph, t5, target): 
        target = self.protein_encoder(target)
        target = torch.cat((graph.ndata['node'], target, t5), 1)
        pdb_graph = self.protein_gt(graph, target)
        x = self.linear1(pdb_graph)
        x = self.relu(x)
        x = self.dropout(x)     
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)     
        x1 = self.linear3(x)
        return x1
    

    