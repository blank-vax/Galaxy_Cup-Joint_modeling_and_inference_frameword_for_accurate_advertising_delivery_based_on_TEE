import pandas as pd
from AFM import Dataset, AFM
from torch.utils.data import DataLoader
import logging
import os

if __name__ == '__main__':
    # logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # logger = logging.getLogger(__name__)
    batch_size = 20000
    
    test_data = Dataset(r'federal_test.csv', batch_size, test=True)
    
    test_gen = DataLoader(test_data, batch_size=batch_size)

    dense_feature_num = 13

    sparse_feature = [1454, 564, 3020973, 814463, 305, 23, 12115, 632, 3, 66113, 5366, 2600906, 3154, 26, 13044, 1814664, 10, 5101, 2116, 4, 2267139, 18, 15, 144827, 102, 89562]

    print("All data received and preprocessed, prediction start")
    model = AFM(dense_feature_num, sparse_feature, embedding_dim=4, gpu='cpu')
    print(model)
    
    model.load_weights('/host/AFM.model')
    result = model.inference(test_gen)
    file = pd.DataFrame()
    file['result'] = result
    file.to_csv('/host/result.csv')