import pandas as pd
from AFM import Dataset, AFM
from torch.utils.data import DataLoader
# import logging
import os

# def fill_na(all_data):
#     for col in all_data.columns:
#         if 'A' in col or 'C' in col:
#             all_data[col] = all_data[col].fillna(0)
#         else:
#             all_data[col] = all_data[col].fillna('')
#     return all_data

# def factorize(all_data):
#     for col in all_data.columns:
#         if 'B' in col or 'D' in col:
#             all_data[col], _ = pd.factorize(all_data[col])
#     return all_data

# def data_split(data, frac=0.1, random_state=2023):
#     """
#     使用重采样将 data 拆分为训练集和验证集
#     Args: data(pandas.DataFrame) 全部已知标签值的训练数据
#           train_path(str) 拆分后的训练数据保存路径
#           valid_path(str) 拆分后的验证数据保存路径
#     Return: merged_data(pandas.DataFrame) 合并后的数据
#     """
#     logger.info('------------------Stage Start------------------')
#     logger.info('Stage: data split')
#     valid_df = data.sample(frac=frac, random_state=random_state, axis=0)
#     train_df = data[~data.index.isin(valid_df.index)]

#     logger.info('train data shape: {}, valid data shape: {}'.format(
#         train_df.shape, valid_df.shape))
#     logger.info('------------------Stage End------------------')
#     return train_df, valid_df


if __name__ == '__main__':
    # logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    # logger.info("[+]Train data loading...")
    # train_A = pd.read_csv(r'train_a.txt')
    # train_B = pd.read_csv(r'train_b.txt')
    # logger.info("[+]Data loading succeed...")
    # logger.info("[+]Train data preprocessing...")
    # # 填补缺失值
    # train_A = fill_na(train_A)
    # train_B = fill_na(train_B)

    # # 离散化处理
    # train_A = factorize(train_A)
    # train_B = factorize(train_B)
    # logger.info("[+]Train data preprocessing succeed...")
    # logger.info("[+]Start characteristics merging...")
    # # 合并
    # all_data = pd.merge(train_A, train_B, left_on=['id'],
    #                         right_on=['id'], how='left')
    # all_data = all_data.drop('id', axis=1)

    # # 调整特征顺序，必需，会影响后续模型训练特征的对应
    # feature_order = ['label', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'C1',
    #     'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'B1', 'B2', 'B3', 'B4', 
    #     'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
    #     'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
    #         'D11', 'D12', 'D13']

    # all_data = all_data[feature_order]
    # # 将所有训练用数据存储于/host/all_data.csv下
    # all_data.to_csv('/host/all_data.csv', index=False)

    # train_df, valid_df = data_split(all_data)
    # logger.info("Store the data in CSV files...")

    # # 执行训练集和验证集分割
    # # 存储于/host/federal_train.csv和/host/federal_valid.csv下
    # train_df.to_csv('/host/federal_train.csv', index=False)
    # valid_df.to_csv('/host/federal_valid.csv', index=False)

    # del train_A, train_B, all_data, train_df, valid_df
    
    
    batch_size = 20000

    # 读取/host/federal_train.csv文件
    
    # train_data = Dataset('/host/federal_train.csv', batch_size)
    # valid_data = Dataset('/host/federal_valid.csv', batch_size)
    print("CSV training files loads")
    train_data = Dataset(r'federal_train.csv', batch_size)
    valid_data = Dataset(r'federal_valid.csv', batch_size)
    
    train_gen = DataLoader(train_data, batch_size=batch_size)
    valid_gen = DataLoader(valid_data, batch_size=batch_size)

    dense_feature_num = 13

    sparse_feature = [1454, 564, 3020973, 814463, 305, 23, 12115, 632, 3, 66113, 5366, 2600906, 3154, 26, 13044, 1814664, 10, 5101, 2116, 4, 2267139, 18, 15, 144827, 102, 89562]

    print("Model constructing...")
    model = AFM(dense_feature_num, sparse_feature, embedding_dim=4, gpu='cpu')
    print(model)
    model.count_parameters()
    epochs = 3

    # 完成模型训练
    print("Model verify")
    model.fit_generator(train_gen, validation_data=valid_gen)
    print("Training finished")