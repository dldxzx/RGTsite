import torch
import os
import numpy as np
from tqdm import tqdm
from bio_embeddings.embed import ProtTransT5XLU50Embedder
T5_EMBEDDER = ProtTransT5XLU50Embedder(half_model=True)

def create_t5_emb(seq):
    embeddings = T5_EMBEDDER.embed(seq)
    return embeddings

strs = ['388', '1930', '1930', 'NW30-TRAIN', 'TEST'] 
for i in range(5):
    # 定义FASTA文件路径
    lab_file = 'RGTsite/lab/PATP-' + strs[i] + '-lab/'
    fasta_file = 'RGTsite/seq/PATP-' + strs[i] + '-seq/'
    # 获取子目录下所有的fasta文件
    seq_files = [f for f in os.listdir(fasta_file) if f.endswith('.fasta')]
    # 遍历每个fasta文件
    for idx, seq_file in enumerate(seq_files):
        print(strs[i] + ': ' + str(idx))
        # 创建文件夹
        tensor_file = 'RGTsite/data/ProtT5'
        if not os.path.exists(tensor_file):
            os.makedirs(tensor_file)
        if not os.path.exists(os.path.join(tensor_file, f"{seq_file[:seq_file.rfind('.')]}.pt")):
            seq_file_path = os.path.join(fasta_file, seq_file)
            lab_file_path = os.path.join(lab_file, seq_file)
            # 打开文件并逐行读取
            with open(seq_file_path, 'r') as f:
                lines = f.readlines()
                sequence = ''.join([line.strip() for line in lines[1:]])  # 假设fasta文件第二行开始是序列
            # 打开文件并逐行读取
            with open(lab_file_path, 'r') as f:
                lines = f.readlines()
                lab = ''.join([line.strip() for line in lines[1:]])  # 假设fasta文件第二行开始是序列
            # 创建一个空列表，用于存储每个序列的嵌入向量
            t5_embeddings = []
            # 遍历每个序列，并生成嵌入向量
            for seq in tqdm(sequence, desc="Generating embeddings"):
                t5_emb = create_t5_emb(seq)
                t5_embeddings.append(t5_emb)
            # 将列表转换为 NumPy 数组，以便处理
            t5_torch_tensor = torch.tensor(np.array(t5_embeddings)).squeeze(axis=1)
            # 合并成一个元组或者字典
            combined_tensors = {'t5': t5_torch_tensor}
            # 保存 PyTorch tensor 到 .pt 文件
            for key, torch_tensor in combined_tensors.items():
                torch.save(combined_tensors, os.path.join(tensor_file, f"{seq_file[:seq_file.rfind('.')]}.pt"))
            # 加载 .pt 文件中的嵌入向量
            loaded_tensor = torch.load(tensor_file + '/' +seq_file[:seq_file.rfind('.')] + '.pt')
            # 输出加载的 tensor 的形状
            print("Loaded tensor shape:", loaded_tensor)
