import dgl
import os
import math
import torch as th
import numpy as np
from Bio.PDB import PDBParser
import numpy as np
from rdkit import Chem
from MDAnalysis.analysis import distances
import MDAnalysis as mda
from itertools import permutations
from scipy.spatial import distance_matrix
import torch
import warnings
warnings.filterwarnings("ignore")


#three letter symbol of amino acid
res_dict ={'ALA':'A', 'CYS':'C', 'ASP':'D', 'GLU':'E', 'PHE':'F', 'GLY':'G', 'HIS':'H', 'ILE':'I', 'LYS':'K', 'LEU':'L',
           'MET':'M', 'ASN':'N', 'PRO':'P', 'GLN':'Q', 'ARG':'R', 'SER':'S', 'THR':'T', 'VAL':'V', 'TRP':'W', 'TYR':'Y'}

pro_res_table = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
                 'X']

METAL = ["LI", "NA", "K", "RB", "CS", "MG", "TL", "CU", "AG", "BE", "NI", "PT", "ZN", "CO", "PD", "AG", "CR", "FE", "V",
         "MN", "HG", 'GA',"CD", "YB", "CA", "SN", "PB", "EU", "SR", "SM", "BA", "RA", "AL", "IN", "TL", "Y", "LA", "CE", 
         "PR", "ND","GD", "TB", "DY", "ER","TM", "LU", "HF", "ZR", "CE", "U", "PU", "TH"]


res_weight_table = {'A': 71.08, 'C': 103.15, 'D': 115.09, 'E': 129.12, 'F': 147.18, 'G': 57.05, 'H': 137.14,
                    'I': 113.16, 'K': 128.18, 'L': 113.16, 'M': 131.20, 'N': 114.11, 'P': 97.12, 'Q': 128.13,
                    'R': 156.19, 'S': 87.08, 'T': 101.11, 'V': 99.13, 'W': 186.22, 'Y': 163.18}

res_pka_table = {'A': 2.34, 'C': 1.96, 'D': 1.88, 'E': 2.19, 'F': 1.83, 'G': 2.34, 'H': 1.82, 'I': 2.36,
                 'K': 2.18, 'L': 2.36, 'M': 2.28, 'N': 2.02, 'P': 1.99, 'Q': 2.17, 'R': 2.17, 'S': 2.21,
                 'T': 2.09, 'V': 2.32, 'W': 2.83, 'Y': 2.32}

res_pkb_table = {'A': 9.69, 'C': 10.28, 'D': 9.60, 'E': 9.67, 'F': 9.13, 'G': 9.60, 'H': 9.17,
                 'I': 9.60, 'K': 8.95, 'L': 9.60, 'M': 9.21, 'N': 8.80, 'P': 10.60, 'Q': 9.13,
                 'R': 9.04, 'S': 9.15, 'T': 9.10, 'V': 9.62, 'W': 9.39, 'Y': 9.62}

res_pkx_table = {'A': 0.00, 'C': 8.18, 'D': 3.65, 'E': 4.25, 'F': 0.00, 'G': 0, 'H': 6.00,
                 'I': 0.00, 'K': 10.53, 'L': 0.00, 'M': 0.00, 'N': 0.00, 'P': 0.00, 'Q': 0.00,
                 'R': 12.48, 'S': 0.00, 'T': 0.00, 'V': 0.00, 'W': 0.00, 'Y': 0.00}

res_hydrophobic_ph2_table = {'A': 47, 'C': 52, 'D': -18, 'E': 8, 'F': 92, 'G': 0, 'H': -42, 'I': 100,
                             'K': -37, 'L': 100, 'M': 74, 'N': -41, 'P': -46, 'Q': -18, 'R': -26, 'S': -7,
                             'T': 13, 'V': 79, 'W': 84, 'Y': 49}

res_hydrophobic_ph7_table = {'A': 41, 'C': 49, 'D': -55, 'E': -31, 'F': 100, 'G': 0, 'H': 8, 'I': 99,
                             'K': -23, 'L': 97, 'M': 74, 'N': -28, 'P': -46, 'Q': -10, 'R': -14, 'S': -5,
                             'T': 13, 'V': 76, 'W': 97, 'Y': 63}


# 该函数的作用是对字典中的数值进行线性归一化，使所有数值都在[0, 1]区间内，并添加一个新的键'X'，该值为字典中最大值和最小值的平均值
def dic_normalize(dic):
    max_value = dic[max(dic, key=dic.get)]
    min_value = dic[min(dic, key=dic.get)]
    interval = float(max_value) - float(min_value)
    for key in dic.keys():
        dic[key] = (dic[key] - min_value) / interval
    dic['X'] = (max_value + min_value) / 2.0
    return dic
    

'''
    这个 residue_features 函数的目的是为蛋白质结构中的一个残基（residue）提取一组特征，可能是用于机器学习或其他数据分析任务。
这些特征来源于三个主要的方面：
        1.残基的化学性质：从归一化的字典数据中提取有关该残基的性质，如酸碱性、疏水性等。
        2.残基的几何距离：基于该残基的原子位置计算的一些几何特性，如最大/最小距离，特定原子之间的距离。
        3.残基的立体化学性质：与残基的旋转角度（如φ、ψ、ω、χ1角度）相关的特征。
'''
def residue_features(res, residues):
    res_property1 = [dic_normalize(res_weight_table)[res], dic_normalize(res_pka_table)[res], 
                     dic_normalize(res_pkb_table)[res], dic_normalize(res_pkx_table)[res],
                     dic_normalize(res_hydrophobic_ph2_table)[res], dic_normalize(res_hydrophobic_ph7_table)[res]] # 6
    res_property2 = [distances.self_distance_array((residues.atoms).positions).max() * 0.1, 
                     distances.self_distance_array((residues.atoms).positions).min() * 0.1, 
                     distances.dist((residues.atoms).select_atoms("name CA"), (residues.atoms).select_atoms("name O"))[-1][0] * 0.1, 
                     distances.dist((residues.atoms).select_atoms("name O"), (residues.atoms).select_atoms("name N"))[-1][0] * 0.1, 
                     distances.dist((residues.atoms).select_atoms("name N"), (residues.atoms).select_atoms("name C"))[-1][0] * 0.1] # 5
    res_property3 = [(residues.phi_selection().dihedral.value() if residues.phi_selection() else 0) * 0.01,
        (residues.psi_selection().dihedral.value() if residues.psi_selection() else 0) * 0.01,
        (residues.omega_selection().dihedral.value() if residues.omega_selection() else 0) * 0.01,
        (residues.chi1_selection().dihedral.value() if residues.chi1_selection() else 0) * 0.01] # 4
    return np.array(res_property1 + res_property2 + res_property3)


'''
    该函数的主要功能是通过计算残基对之间的几何距离和相似性，来构建一个包含边的图结构，其中每条边代表两个残基之间的空间关系。
    这个函数对于蛋白质结构分析或图形化表示蛋白质的空间关系（如蛋白质折叠、交互作用分析等）非常有用。
'''
def obatin_edge(u, cutoff=8.0):
    edgeids = []
    dismin = []
    dismax = []
    dissim = []
    # 遍历所有残基对 (res1, res2)，生成所有两个残基的排列组合
    for res1, res2 in permutations(u.residues, 2):  
        # 获取res1中所有原子的坐标，转为NumPy数组并展平
        cos1 = np.array(res1.atoms.positions).flatten()  
        # 获取res2中所有原子的坐标，转为NumPy数组并展平
        cos2 = np.array(res2.atoms.positions).flatten()  
        # 计算两个坐标数组的最小长度（以保证后续计算的有效性）
        min_length = min(cos1.shape[0], cos2.shape[0])  
        # 计算两个坐标数组的余弦相似度
        sim = cos1[:min_length].dot(cos2[:min_length]) / (np.linalg.norm(cos1[:min_length])*np.linalg.norm(cos2[:min_length]))  
        # 计算两个残基之间的原子距离
        dist = distances.distance_array(res1.atoms.positions, res2.atoms.positions)  
        # 如果最小距离小于或等于指定阈值，则认为这两个残基有接触
        if dist.min() <= cutoff:  
            # 将符合条件的残基对索引添加到edgeids中
            edgeids.append([res1.ix, res2.ix])  
            # 将最小距离乘以0.1后加入dismin列表
            dismin.append(dist.min() * 0.1)  
            # 将最大距离乘以0.1后加入dismax列表
            dismax.append(dist.max() * 0.1)  
            # 将余弦相似度加入dissim列表
            dissim.append(sim)  
    # 返回包含残基对信息的edgeids以及一个包含最小距离、最大距离、余弦相似度的数组
    return edgeids, np.array([dismin, dismax, dissim]).T  



# 定义一个函数 obtain_ca_pos，用于获取给定残基 res 的Cα原子坐标
def obtain_ca_pos(res):
    # 如果残基的名称前两个字符是 "CA", "FE", 或 "CU"，则将 resname 设置为该前两字符
    if res.resname[:2] in ["CA", "FE", "CU"]:
        resname = res.resname[:2]
    else:
        # 否则，去掉残基名称前后的空白字符，将其赋值给 resname
        resname = res.resname.strip()
    # 如果 resname 是 "M"，直接返回该残基的第一个原子的坐标, 否则，如果该残基有Cα原子（"name CA"），
    # 则返回第一个Cα原子的坐标,如果没有Cα原子，返回所有原子的坐标的均值（即残基原子的平均坐标）
    return res.atoms.positions[0] if resname == "M" else \
       res.atoms.select_atoms("name CA").positions[0] if len(res.atoms.select_atoms("name CA")) > 0 else \
       res.atoms.positions.mean(axis=0)
       

def TargetToGraph(prot_pdb):
    # 使用PDBParser解析蛋白质结构文件
    parser = PDBParser()
    structure = parser.get_structure('protein', prot_pdb)
    # 获取第一个模型（通常是唯一的模型）
    model = structure[0]
    # 提取氨基酸序列和C-alpha原子的坐标
    seq3 = ""
    # 遍历结构中的每个模型、链和残基
    for model in structure:
        for chain in model:
            for residue in chain:
                # 只考虑非水分子和有效的氨基酸残基
                if residue.get_id()[0] == ' ' and residue.get_resname() != "HOH":
                    # 获取氨基酸的三字母代码并添加到序列中
                    seq3 += residue.get_resname()       
    '''
        这行代码通过 RDKit 的 Chem.MolFromPDBFile 函数解析蛋白质的 PDB 文件，并使用 MDAnalysis 的 Universe 类创建一个分子动力学系统对象。
        MolFromPDBFile 参数的意义如下：sanitize=True表示在加载时自动检查并修复分子结构问题，removeHs=True会去除氢原子以简化结构，flavor=0 
        决定氢原子类型的处理方式，proximityBonding=False 禁止自动添加缺失的化学键。最终，MDAnalysis 的 Universe 对象将通过解析 RDKit 生成
        的分子对象来进行后续的分子动力学分析。
    '''
    u = mda.Universe(Chem.MolFromPDBFile(prot_pdb, sanitize=True, removeHs=True, flavor=0, proximityBonding=False))
    # 创建了一个空的图DGL，它将用于存储节点和边的关系
    G = dgl.DGLGraph()
    # 获取蛋白质的所有残基数目，然后将这些残基作为节点添加到DGL图中
    G.add_nodes(len(u.residues))       
    
    # 节点特征 21 + 6 + 5 + 4
    # 通过每3个字符（即每一个氨基酸的三字母编码）提取序列，并将其转换为所需的格式（通过 res_dict），最后构建出新的 seq 字符串
    seq = ''
    for i in range(0, len(seq3), 3):
        triplet = seq3[i:i+3]
        converted_triplet = res_dict[triplet]
        seq += converted_triplet
    # 蛋白质序列中的每个氨基酸生成一个独热编码
    pro_hot = np.array([[seq[i] == s for s in pro_res_table] for i in range(len(seq))], dtype=np.float32)
    # 为每个氨基酸计算属性特征。计算每个氨基酸的化学性质、几何距离和立体化学性质
    pro_property = np.array([residue_features(seq[i], u.residues[i]) for i in range(len(seq))], dtype=np.float32)
    # 氨基酸的类别编码和它的化学性质、几何距离和立体化学性质属性，node_features 是最终的节点特征矩阵，它将作为DGL图中每个节点的特征
    node_features = np.concatenate((pro_hot, pro_property), axis=1)    
    
    # 边特征 6
    # 获取图中的边（节点对）和边的距离矩阵，最大距离为8.0
    edgeids, distm = obatin_edge(u, 8.0)
    # 将边的节点对（源节点，目标节点）分离为两个列表，分别存储源节点和目标节点
    src_list, dst_list = zip(*edgeids)
    # 在图G中添加边，使用源节点和目标节点列表
    G.add_edges(src_list, dst_list)
    # 为图中的节点添加特征，转换为Tensor格式并设为浮点类型
    G.ndata['node'] = th.from_numpy(np.array(node_features)).to(th.float32) 
    # 获取每个氨基酸残基的Cα原子位置并转换为Tensor
    ca_pos = torch.tensor(np.array([obtain_ca_pos(res) for res in u.residues]))
    # 获取每个氨基酸残基的质心位置并转换为Tensor
    center_pos = torch.tensor(u.atoms.center_of_mass(compound='residues'))
    # 计算Cα原子位置之间的距离矩阵
    ca_pos_matx = distance_matrix(ca_pos, ca_pos)
    # 计算质心之间的距离矩阵
    center_pos_matx = distance_matrix(center_pos, center_pos)
    # 计算每对边（源节点，目标节点）之间Cα原子的距离并缩放（0.1）
    cadist = torch.tensor([ca_pos_matx[i, j] for i, j in edgeids]) * 0.1
    # 计算每对边（源节点，目标节点）之间质心的距离并缩放（0.1）
    cedist = torch.tensor([center_pos_matx[i, j] for i, j in edgeids]) * 0.1
    # 计算边连接类型，判断是否为相邻残基的连接（连接类型为1），否则为0
    edge_connect = torch.tensor(np.array([1 if abs(x - y) == 1 and (len(u.residues[min(x, y):min(x, y) + 2].get_connections  \
        ("bonds")) == len(u.residues[min(x, y)].get_connections("bonds")) + len(u.residues[min(x, y) + 1].get_connections  \
        ("bonds")) - 1) else 0 for x, y in zip(src_list, dst_list)]))
    # 为图中的边添加特征，包括连接类型、Cα距离、质心距离和其他计算得到的距离
    G.edata["edge"] = torch.cat([edge_connect.view(-1, 1), cadist.view(-1, 1), cedist.view(-1, 1), torch.tensor(distm)], dim=1)
    return G

            
strs = ['388', 'NW30-TEST', 'TEST', '1930', 'NW30-TRAIN'] 
for i in range(5):
    # 定义FASTA文件路径
    pdb_path = 'RGTsite/data/Graph/'
    # 获取子目录下所有的fasta文件
    pdb_files = [f for f in os.listdir(pdb_path) if f.endswith('.pdb')]
    # 遍历每个fasta文件
    for idx, pdb_file in enumerate(pdb_files):
        pdb = 'RGTsite/data/pdb/'
        G = TargetToGraph(os.path.join(pdb_path, pdb_file))
        if not os.path.exists(pdb):
            os.makedirs(pdb)
        torch.save(G, pdb + pdb_file[:pdb_file.rfind('.')] + '.pt')
        # 加载 .pt 文件中的嵌入向量
        loaded_tensor = torch.load(pdb + pdb_file[:pdb_file.rfind('.')] + '.pt')
        # 输出加载的 tensor 的形状
        print("Loaded tensor shape:", loaded_tensor)
