from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import torch
import os
import dgl
from tqdm import tqdm
import numpy as np
from src.asl import  AsymmetricLoss
import torch.nn as nn
from sklearn.metrics import confusion_matrix, roc_auc_score
from src.model import DTS_GraphTransformer


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
        self.data, self.slices = torch.load(self.processed_paths[0]) 
        
    @property
    def processed_file_names(self):
        return [self.processed_data]
    
    def process(self):
        pass
      
#训练模型
def train(loader):
    model.train()
    total_loss = 0
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    for idx, data in loader:
        optimizer.zero_grad()
        # Target CNN
        target = torch.LongTensor([seqs2int(data.target[0])])
        out = model((dgl.batch(data.graph)).to(device), data.t5.to(device), target.to(device))
        # 将输入数据传入模型进行前向传播
        label = torch.tensor([float(bit) for bit in data.y[0]], dtype=torch.float)
        loss_asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)
        loss = loss_asl(out.flatten(), label.to(device))  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        # 二分类
        m = nn.Sigmoid()
        pred1 = (m(out) >= 0.5).float()
        # 将每个字符转换为整数
        label = torch.tensor([int(bit) for bit in data.y[0]])
        true_labels.extend(label.tolist())
        predicted_labels.extend(pred1.tolist())
        predicted_probs.extend(m(out).tolist())
    # 计算tn, fp, fn, tp
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).flatten()
    # 计算 Spe
    Spe = tn / (tn + fp)
    # 计算 Sen 
    Sen = tp / (tp + fn)
    # 计算 Pre
    Pre = tp / (tp + fp)
    # 计算 Acc
    Acc = (tn + tp) / (tn + fp + tp + fn)
    # 计算 F1
    F1 = (2 * Sen * Pre) / (Sen + Pre)
    # 计算 MCC
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    MCC = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # 计算AUC
    auc = roc_auc_score(true_labels, predicted_probs)
    return total_loss / len(loop), Sen, Spe, Acc, Pre, F1, MCC, tn, fp, tp, fn, auc


#训练模型
def val(loader):
    model.eval()
    true_labels = []
    total_loss = 0
    predicted_labels = []
    predicted_probs = []
    with torch.no_grad():
        for idx, data in loader:
            optimizer.zero_grad()
            # Target CNN
            target = torch.LongTensor([seqs2int(data.target[0])])
            out = model((dgl.batch(data.graph)).to(device), data.t5.to(device), target.to(device))
            # 将输入数据传入模型进行前向传播
            label = torch.tensor([float(bit) for bit in data.y[0]], dtype=torch.float)
            loss_asl = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True).to(device)
            loss = loss_asl(out.flatten(), label.to(device))    
            total_loss += loss.item()
            # 二分类
            m = nn.Sigmoid()
            pred1 = (m(out) >= 0.5).float()
            # 将每个字符转换为整数
            label = torch.tensor([int(bit) for bit in data.y[0]])
            true_labels.extend(label.tolist())
            predicted_labels.extend(pred1.tolist())
            predicted_probs.extend(m(out).tolist())
    # 计算tn, fp, fn, tp
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).flatten()
    # 计算 Spe
    Spe = tn / (tn + fp)
    # 计算 Sen 
    Sen = tp / (tp + fn)
    # 计算 Pre
    Pre = tp / (tp + fp)
    # 计算 Acc
    Acc = (tn + tp) / (tn + fp + tp + fn)
    # 计算 F1
    F1 = (2 * Sen * Pre) / (Sen + Pre)
    # 计算 MCC
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    MCC = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # 计算AUC
    auc = roc_auc_score(true_labels, predicted_probs)
    return total_loss / len(loop), Sen, Spe, Acc, Pre, F1, MCC, tn, fp, tp, fn, auc


# 测试模型，并计算Top1-10指标
def test(loader):
    model.eval()
    true_labels = []
    predicted_labels = []
    predicted_probs = []
    with torch.no_grad():
        for idx, data in loader:
            optimizer.zero_grad()
            # Target CNN
            target = torch.LongTensor([seqs2int(data.target[0])])
            out = model((dgl.batch(data.graph)).to(device), data.t5.to(device), target.to(device))
            # 二分类
            m = nn.Sigmoid()
            pred1 = (m(out) >= 0.5).float()
            # 将每个字符转换为整数
            label = torch.tensor([int(bit) for bit in data.y[0]])
            true_labels.extend(label.tolist())
            predicted_labels.extend(pred1.tolist())
            predicted_probs.extend(m(out).tolist())
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)
    predicted_probs = np.array(predicted_probs)
    tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).flatten()
    # 计算 Spe
    Spe = tn / (tn + fp)
    # 计算 Sen 
    Sen = tp / (tp + fn)
    # 计算 Pre
    Pre = tp / (tp + fp)
    # 计算 Acc
    Acc = (tn + tp) / (tn + fp + tp + fn)
    # 计算 F1
    F1 = (2 * Sen * Pre) / (Sen + Pre)
    # 计算 MCC
    tp = float(tp)
    tn = float(tn)
    fp = float(fp)
    fn = float(fn)
    MCC = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    # 计算AUC
    auc = roc_auc_score(true_labels, predicted_probs)
    return Sen, Spe, Acc, Pre, F1, MCC, tn, fp, tp, fn, auc


if __name__ == '__main__':
    train_root = 'RGTsite/data/train'
    test_root = 'RGTsite/data/test'
    val_root = 'RGTsite/data/val'
    batch_size = 1
    tt = 10
    epochs = 200
    loss_min = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DTS_GraphTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-5, eps=1e-7, betas=(0.9, 0.999))
    train_dataset = SMILES_Protein_Dataset(root=train_root,raw_dataset='train_388.csv',processed_data='train_388.pt')
    val_dataset = SMILES_Protein_Dataset(root=val_root,raw_dataset='val_388.csv',processed_data='val_388.pt')
    test_dataset = SMILES_Protein_Dataset(root=test_root,raw_dataset='test.csv',processed_data='test.pt')
    # 打开文件，以追加方式写入
    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}")
        train_dataloader = DataLoader(train_dataset,batch_size,shuffle=True,drop_last=False)
        test_dataloader = DataLoader(test_dataset,batch_size,shuffle=False,drop_last=False)
        val_dataloader = DataLoader(val_dataset,batch_size,shuffle=False,drop_last=False)
        with open(f"RGTsite/results/388.txt", 'a') as f:
            loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), colour='red', desc='Train Loss')
            loss, Sen, Spe, Acc, Pre, F1, MCC, tn, fp, tp, fn, auc = train(loop)
            print(loss, Sen, Spe, Acc, Pre, F1, MCC, auc)    
            f.write('Epoch {:03d}, Loss: {:.4f}\n'.format(epoch+1, loss))
            f.write('Train:  TN: {:}, FP: {:}, TP: {:}, FN: {:}\n'.format(tn, fp, tp, fn))
            f.write('Train:  Sen: {:.4f}, Spe: {:.4f}, Acc: {:.4f}, Pre: {:.4f}, F1: {:.4f}, MCC: {:.4f}, AUC: {:.4f}\n'
                    .format(Sen, Spe, Acc, Pre, F1, MCC, auc))
            loop = tqdm(enumerate(val_dataloader), total=len(val_dataloader), colour='red', desc='Val')
            loss, Sen, Spe, Acc, Pre, F1, MCC, tn, fp, tp, fn, auc = val(loop)
            print(loss, Sen, Spe, Acc, Pre, F1, MCC)   
            if loss_min > loss:
                loss_min = loss
                sum = 0
            else:
                sum += 1
            if sum >= tt:
                break
            f.write('Val:  TN: {:}, FP: {:}, TP: {:}, FN: {:}\n'.format(tn, fp, tp, fn))
            f.write('Val:  Sen: {:.4f}, Spe: {:.4f}, Acc: {:.4f}, Pre: {:.4f}, F1: {:.4f}, MCC: {:.4f}, AUC: {:.4f}\n'
                    .format(Sen, Spe, Acc, Pre, F1, MCC, auc))
            loop = tqdm(enumerate(test_dataloader), total=len(test_dataloader), colour='red', desc='Test')
            Sen, Spe, Acc, Pre, F1, MCC, tn, fp, tp, fn, auc = test(loop)
            print(Sen, Spe, Acc, Pre, F1, MCC)   
            f.write('Test:  TN: {:}, FP: {:}, TP: {:}, FN: {:}\n'.format(tn, fp, tp, fn))
            f.write('Test:  Sen: {:.4f}, Spe: {:.4f}, Acc: {:.4f}, Pre: {:.4f}, F1: {:.4f}, MCC: {:.4f}, AUC: {:.4f}\n'
                    .format(Sen, Spe, Acc, Pre, F1, MCC, auc))
            model_path = f'RGTsite/results/388/'
            # 检查目录是否存在
            if not os.path.exists(model_path):
                # 如果目录不存在，则创建它
                os.makedirs(model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(f"{model_path}{epoch+1}.pkl"))   
        # 关闭文件
        f.close()
    
    