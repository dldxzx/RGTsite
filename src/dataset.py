from torch_geometric.data import InMemoryDataset, Data
import torch
import os
import numpy as np
import pandas as pd


VOCAB_PROTEIN = {"A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6,
                 "H": 7, "I": 8, "K": 9, "L": 10, "M": 11, "N": 12,
                 "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
                 "W": 19, "Y": 20, "X": 21}

NUM_TO_LETTER = {v:k for k, v in VOCAB_PROTEIN.items()}

def seqs2int(target):
    return [VOCAB_PROTEIN[s] for s in target]

class SMILES_Protein_Dataset(InMemoryDataset):
    def __init__(self, root, raw_dataset=None, processed_data=None, transform = None, pre_transform = None):
        self.root=root
        self.raw_dataset=raw_dataset
        self.processed_data=processed_data
        self.max_smiles_len=256
        self.smiles_dict_len=65
        super(SMILES_Protein_Dataset,self).__init__(root, transform, pre_transform)
        if os.path.exists(self.processed_paths[0]):
            self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            # self.process()
            self.data, self.slices = torch.load(self.processed_paths[0])
        
    # 原始文件位置
    @property
    def raw_file_names(self):
        return [self.raw_dataset]
    
    # 文件保存位置
    @property
    def processed_file_names(self):
        return [self.processed_data]
        # return []
    
    def download(self):
        pass
        
    def process(self):
        # 读取CSV文件
        data = pd.read_csv(self.raw_paths[0])
        # 使用字符串操作获取文件名部分
        file_name = self.raw_paths[0].split("/")[-1]
        bio_embeddings_path = 'RGTsite/data/ProtT5/'
        graph_path = 'RGTsitew/data/Graph/'
        data_list = []
        for index, row in data.iterrows():
            print(file_name[:-4] + " : " + str(index+1))
            # 获取每一行数据
            target_name = row['target_name']  # 确保target_name是字符串类型
            target = row['target']
            label = row['label']
            bio_embeddings_file = os.path.join(bio_embeddings_path, target_name[:target_name.rfind('.')] + '.pt')
            graph_file = os.path.join(graph_path, target_name[:target_name.rfind('.')] + '.pt')
            # 加载 .pt 文件中的嵌入向量
            bert_torch_tensor = torch.load(bio_embeddings_file)['bert']
            t5_torch_tensor = torch.load(bio_embeddings_file)['t5']
            graph = torch.load(graph_file)
            # 保存特征成.pt文件
            data = Data(graph = graph, bert = bert_torch_tensor, t5 = t5_torch_tensor, target = target, y = label, target_name = target_name)
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data,slices), self.processed_paths[0]) 


if __name__ == '__main__':
    train_root = 'RGTsite/data/train'
    test_root = 'RGTsite/data/test'
    val_root = 'RGTsite/data/val'
    
    train_dataset = SMILES_Protein_Dataset(root=train_root,raw_dataset='train_388.csv',processed_data='train_388.pt')
    train_dataset = SMILES_Protein_Dataset(root=train_root,raw_dataset='train_1930.csv',processed_data='train_1930.pt')
    train_dataset = SMILES_Protein_Dataset(root=train_root,raw_dataset='nw30_train.csv',processed_data='nw30_train.pt')
