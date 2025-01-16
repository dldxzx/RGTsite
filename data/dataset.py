import pandas as pd
import os


'''  
    这段代码定义了一个函数 save_sequences_from_fasta，用于从给定的FASTA文件中提取每个序列，并将它们保存为独立的文件。
每个新文件的文件名是基于序列的头部（即以 > 开头的行），并且文件内容包括头部信息和对应的序列。
    具体步骤如下：
        1.读取输入的FASTA文件。
        2.当遇到以 > 开头的行时，提取头部信息并创建新的文件，文件名基于头部。
        3.将序列（非头部行）累加到当前序列中。
        4.在遇到新的头部或文件结束时，将当前序列保存到新文件。
        5.最后处理最后一个序列并保存。
    该函数的作用是将FASTA文件中的多个序列分割成独立的文件，文件名基于序列头部生成，每个文件包含对应的头部和序列内容。
通过调用该函数，可以将多个FASTA文件中的序列提取并保存为多个新的文件，存放在指定的输出目录中。
'''
def save_sequences_from_fasta(input_file, out_file):
    with open(input_file, 'r') as f:
        current_sequence = ""
        current_header = None
        for line in f:
            if line.startswith('>'):
                # 如果有当前序列，保存到文件中
                if current_sequence and current_header:
                    filename = current_header[1:].strip() + '.fasta'  # 使用头部名称命名文件
                    with open(out_file + filename, 'w') as out:
                        out.write(current_header + '\n')
                        out.write(current_sequence + '\n')
                        print(filename)
                        print(current_sequence)
                # 更新当前头部和序列
                current_header = line.strip()
                current_sequence = ""
            else:
                current_sequence += line.strip()
        
        # 处理最后一个序列
        if current_sequence and current_header:
            filename = current_header[1:].strip() + '.fasta'  # 使用头部名称命名文件
            with open(out_file + filename, 'w') as out:
                out.write(current_header + '\n')
                out.write(current_sequence + '\n')

# 调用函数并传入FASTA文件路径
# PATP-388-seq
save_sequences_from_fasta('RGTsite/data/PATP-388_seq.fa', 'RGTsite/data/seq/PATP-388-seq/')
# PATP-NW30-TRAIN_seq
save_sequences_from_fasta('RGTsite/data/PATP-1930_seq.fa', 'RGTsite/data/seq/PATP-1930-seq/')
# PATP-NW30-TRAIN_seq
save_sequences_from_fasta('RGTsite/data/PATP-NW30-TRAIN_seq.fa', 'RGTsite/data/seq/PATP-NW30-TRAIN-seq/')
# PATP-NW30-TEST_seq
save_sequences_from_fasta('RGTsite/data/PATP-NW30-TEST_seq.fa', 'RGTsite/data/seq/PATP-NW30-TEST-seq/')
# PATP-TEST_seq
save_sequences_from_fasta('RGTsite/data/PATP-TEST_seq.fa', 'RGTsite/data/seq/PATP-TEST-seq/')
# PATP-388-lab
save_sequences_from_fasta('RGTsite/data/PATP-388_lab.fa', 'RGTsite/data/lab/PATP-388-lab/')
# PATP-NW30-TRAIN_lab
save_sequences_from_fasta('RGTsite/data/PATP-1930_lab.fa', 'RGTsite/data/lab/PATP-1930-lab/')
# PATP-NW30-TRAIN_lab
save_sequences_from_fasta('RGTsite/data/PATP-NW30-TRAIN_lab.fa', 'RGTsite/data/lab/PATP-NW30-TRAIN-lab/')
# PATP-NW30-TEST_lab
save_sequences_from_fasta('RGTsite/data/PATP-NW30-TEST_lab.fa', 'RGTsite/data/lab/PATP-NW30-TEST-lab/')
# PATP-TEST_lab
save_sequences_from_fasta('RGTsite/data/PATP-TEST_lab.fa', 'RGTsite/data/lab/PATP-TEST-lab/')



'''
    这段代码的功能是从指定路径下的FASTA文件和对应的标签文件中读取序列数据，并将其整理成CSV格式。具体来说：
        1.数据处理：程序首先在两个不同的循环中分别处理训练集和测试集数据。训练集处理的文件名由strs数组中的元素定义，而
测试集处理的文件名则由第二个strs数组定义。每个训练集或测试集文件夹下，有对应的FASTA文件（包含序列数据）和标签文件。程序
从这些文件中读取序列数据和标签数据。假设FASTA文件的第二行开始是实际的序列数据，标签文件的处理方法类似。
        2.数据存储：每个FASTA文件的内容（序列）和标签被整理成字典形式（target_name, target, label）。然后，将所有数据
保存在一个Pandas DataFrame中，并将其导出为CSV文件，存储路径和文件名分别由strs1数组和目标文件夹结构指定。
        3.输出结果：程序将训练集数据保存为RGTsite/data/train/raw/文件夹中的CSV文件，将测试集数据保存为RGTsite/data/test/raw/文件夹中的CSV文件。
    总的来说，代码的作用是将FASTA文件和标签文件中的数据提取、整理并存储为CSV文件，以供后续的数据处理和模型训练使用。
'''
strs = ['388', '1930', 'NW30-TRAIN'] 
strs1 = ['train_388', 'train_1930', 'nw30_train'] 
for k in range(3):
    data_list = []
    # 定义FASTA文件路径
    lab_file = 'RGTsite/lab/PATP-' + strs[k] + '-lab/'
    fasta_file = 'RGTsite/seq/PATP-' + strs[k] + '-seq/'
    # 获取子目录下所有的fasta文件
    seq_files = [f for f in os.listdir(fasta_file) if f.endswith('.fasta')]
    # 创建一个空的DataFrame来存储数据
    data = []
    # 遍历每个fasta文件
    for seq_file in seq_files:
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
        # 将数据插入DataFrame
        data.append({'target_name': seq_file, 'target': sequence, 'label': lab})
    # 将数据列表转换为DataFrame
    df = pd.DataFrame(data)
    # 将DataFrame写入CSV文件
    df.to_csv(f'RGTsite/data/train/raw/{strs1[k]}.csv', index=False)

strs = ['NW30-TEST', 'TEST'] 
strs = ['nw30_test', 'test'] 
for k in range(2):
    data_list = []
    # 定义FASTA文件路径
    lab_file = 'RGTsite/lab/PATP-' + strs[k] + '-lab/'
    fasta_file = 'RGTsite/seq/PATP-' + strs[k] + '-seq/'
    # 获取子目录下所有的fasta文件
    seq_files = [f for f in os.listdir(fasta_file) if f.endswith('.fasta')]
    # 创建一个空的DataFrame来存储数据
    data = []
    # 遍历每个fasta文件
    for seq_file in seq_files:
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
        # 将数据插入DataFrame
        data.append({'target_name': seq_file, 'target': sequence, 'label': lab})
    # 将数据列表转换为DataFrame
    df = pd.DataFrame(data)
    # 将DataFrame写入CSV文件
    df.to_csv(f'RGTsite/data/test/raw/{strs1[k]}.csv', index=False)