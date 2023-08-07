"""
数据预处理阶段, machineA和machineB两方均会进行数据预处理
"""
import pandas as pd
import logging
def fill_na(all_data):
    for col in all_data.columns:
        if 'A' in col or 'C' in col:
            all_data[col] = all_data[col].fillna(0)
        else:
            all_data[col] = all_data[col].fillna('')
    return all_data

def factorize(all_data):
    for col in all_data.columns:
        if 'B' in col or 'D' in col:
            all_data[col], _ = pd.factorize(all_data[col])
    return all_data

def data_split(data, frac=0.1, random_state=2023):
    """
    使用重采样将 data 拆分为训练集和验证集
    Args: data(pandas.DataFrame) 全部已知标签值的训练数据
          train_path(str) 拆分后的训练数据保存路径
          valid_path(str) 拆分后的验证数据保存路径
    Return: merged_data(pandas.DataFrame) 合并后的数据
    """
    logger.info('[+] Data split stage start')
    logger.info('[+] Stage: data split')
    valid_df = data.sample(frac=frac, random_state=random_state, axis=0)
    train_df = data[~data.index.isin(valid_df.index)]

    logger.info('[+] Train data shape: {}, valid data shape: {}'.format(
        train_df.shape, valid_df.shape))
    logger.info('[+] Data split stage end')
    return train_df, valid_df

if __name__ == '__main__':
    # 配置日志相关文件
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)