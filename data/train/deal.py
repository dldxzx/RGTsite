import pandas as pd

# 读取两个CSV文件
file1 = '/home/user/sgp/Drug_Target_Sites/DTS/data/train/raw/train_388.csv'  # 替换为你的第一个文件路径
file2 = '/home/user/sgp/Drug_Target_Sites/DTS/data/test/raw/test.csv'  # 替换为你的第二个文件路径
# 加载CSV文件到DataFrame
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 假设你要检查的列名都是'target_name'
# 判断df1的target_name是否在df2的target_name中
df1['exists_in_df2'] = df1['target_name'].isin(df2['target_name'])
# 输出结果，查看exists_in_df2列是否有True
if df1['exists_in_df2'].any():
    # print("388存在的target_name:")
    exists_df = df1[df1['exists_in_df2']]
    # print(exists_df)  # 打印出所有存在的target_name
    print(f"388存在的target_name数量: {exists_df.shape[0]}")  # 输出存在的target_name的数量
else:
    print("388没有存在的target_name。")
    

# 读取两个CSV文件
file1 = '/home/user/sgp/Drug_Target_Sites/DTS/data/train/raw/train_1930.csv'  # 替换为你的第一个文件路径
file2 = '/home/user/sgp/Drug_Target_Sites/DTS/data/test/raw/test.csv'  # 替换为你的第二个文件路径
# 加载CSV文件到DataFrame
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 假设你要检查的列名都是'target_name'
# 判断df1的target_name是否在df2的target_name中
df1['exists_in_df2'] = df1['target_name'].isin(df2['target_name'])
# 输出结果，查看exists_in_df2列是否有True
if df1['exists_in_df2'].any():
    # print("1930存在的target_name:")
    exists_df = df1[df1['exists_in_df2']]
    # print(exists_df)  # 打印出所有存在的target_name
    print(f"1930存在的target_name数量: {exists_df.shape[0]}")  # 输出存在的target_name的数量
else:
    print("1930没有存在的target_name。")
    
    
# 读取两个CSV文件
file1 = '/home/user/sgp/Drug_Target_Sites/DTS/data/train/raw/nw30_train.csv'  # 替换为你的第一个文件路径
file2 = '/home/user/sgp/Drug_Target_Sites/DTS/data/test/raw/nw30_test.csv'  # 替换为你的第二个文件路径
# 加载CSV文件到DataFrame
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 假设你要检查的列名都是'target_name'
# 判断df1的target_name是否在df2的target_name中
df1['exists_in_df2'] = df1['target_name'].isin(df2['target_name'])
# 输出结果，查看exists_in_df2列是否有True
if df1['exists_in_df2'].any():
    # print("nw30存在的target_name:")
    exists_df = df1[df1['exists_in_df2']]
    # print(exists_df)  # 打印出所有存在的target_name
    print(f"nw30存在的target_name数量: {exists_df.shape[0]}")  # 输出存在的target_name的数量
else:
    print("nw30没有存在的target_name。")
        