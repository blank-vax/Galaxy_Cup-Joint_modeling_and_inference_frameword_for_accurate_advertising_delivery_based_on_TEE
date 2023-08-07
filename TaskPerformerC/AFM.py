import os
import sys
import torch
from torch import nn
from torch.utils import data
import pandas as pd
import numpy as np
from itertools import combinations # python内置库中计算组合的方法
from tqdm import tqdm # 进度条组件
from metrics import evaluate_metrics
import logging

class Dataset(data.Dataset):
    def __init__(self, file_path, batch_size, test = False):
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.df = pd.read_csv(file_path, nrows=batch_size)
        self.darray = self.df.to_numpy()
        self.iter_flag = 0
        self.file_path = file_path
        self.batch_size = batch_size
        self.test = test

    def __getitem__(self, index):
        if index // self.batch_size != self.iter_flag:
            self.iter_flag = index // self.batch_size
            self.df = pd.read_csv(self.file_path, skiprows=self.iter_flag*self.batch_size, nrows=self.batch_size)
            self.darray = self.df.to_numpy()
        if self.test:
            return self.darray[index-self.iter_flag*self.batch_size, 1:], 0
        X = self.darray[index-self.iter_flag*self.batch_size, 1:]
        y = self.darray[index-self.iter_flag*self.batch_size, 0]
        return X, y

    def __len__(self):
        if 'train' in self.file_path:
            return 9010911
        if 'valid' in self.file_path:
            return 1001212
        if self.test:
            return 1507594
        return 9010911 + 1001212

class AFM(nn.Module):
    def __init__(self,
                 dense_feature_num,
                 sparse_feature,
                 gpu=0,
                 learning_rate=1e-3,
                 embedding_dim=10,
                 attention_dropout=[0, 0],
                 attention_dim=10,
                 early_stopping_patience=2):
        super(AFM, self).__init__()
        logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        self.dense_feature_num = dense_feature_num
        self.sparse_feature_num = len(sparse_feature)
        self.device = gpu
        self.embedding_dim = embedding_dim
        self._validation_metrics = ['logloss', 'AUC']
        self._best_metric = -np.Inf
        # AFM.model文件存储于/host/下
        self.checkpoint = "/host/AFM.model"
        # self.checkpoint = os.path.abspath(os.path.join(os.curdir, "AFM.model")) # 存到当前文件夹下，命名为 AFM.model
        self._patience = early_stopping_patience

        self.dense_embedding = nn.ModuleDict({'linear_{}'.format(i): nn.Linear(1, embedding_dim, bias=False) for i in range(self.dense_feature_num)})
        self.sparse_embedding = nn.ModuleDict({'emb_{}'.format(i): nn.Embedding(num_embeddings=feat, embedding_dim=embedding_dim)\
                                                for i, feat in enumerate(sparse_feature)})

        p, q = zip(*list(combinations(range(self.sparse_feature_num), 2)))
        self.field_p = nn.Parameter(torch.LongTensor(p), requires_grad=False)
        self.field_q = nn.Parameter(torch.LongTensor(q), requires_grad=False)

        self.attention = nn.Sequential(nn.Linear(embedding_dim, attention_dim),
                                       nn.ReLU(),
                                       nn.Linear(attention_dim, 1, bias=False),
                                       nn.Softmax(dim=1))
        self.weight_p = nn.Linear(embedding_dim, 1, bias=False)
        self.dropout1 = nn.Dropout(attention_dropout[0])
        self.dropout2 = nn.Dropout(attention_dropout[1])
        self.lr_layer = nn.Linear(embedding_dim, 1)

        self.output_activation = nn.Sigmoid()

        self.optimizer = torch.optim.Adam(self.parameters(), learning_rate)
        self.loss_fn = nn.BCELoss()
        self.to(device=self.device)

    def forward(self, inputs):
        X, y = self._inputs_to_device(inputs)
        dense_feature = X[:, :self.dense_feature_num]
        sparse_feature = X[:, self.dense_feature_num:]

        dense_feature = [self.dense_embedding['linear_{}'.format(i)](dense_feature[:, i]) for i in range(self.dense_feature_num)]
        dense_feature = torch.stack(dense_feature).reshape(-1, self.dense_feature_num, self.embedding_dim)
        sparse_feature = [self.sparse_embedding['emb_{}'.format(i)](sparse_feature[:, i].int())\
                        for i in range(self.sparse_feature_num)]
        sparse_feature = torch.stack(sparse_feature).reshape(-1, self.sparse_feature_num, self.embedding_dim)

        # 交叉部分
        emb1 =  torch.index_select(sparse_feature, 1, self.field_p)
        emb2 = torch.index_select(sparse_feature, 1, self.field_q)
        elementwise_product = emb1 * emb2

        # Attention 部分
        attention_weight = self.attention(elementwise_product)
        attention_weight = self.dropout1(attention_weight)
        attention_sum = torch.sum(attention_weight * elementwise_product, dim=1)
        attention_sum = self.dropout2(attention_sum)
        afm_out = self.weight_p(attention_sum)

        # 特征加总
        y_pred = self.lr_layer(dense_feature.sum(dim=1)) + afm_out

        y_pred = self.output_activation(y_pred)
        return_dict = {"y_true": y, "y_pred": y_pred}
        return return_dict

    def _inputs_to_device(self, inputs):
        X, y = inputs
        X = X.unsqueeze(-1).to(torch.float32).to(self.device)
        y = y.float().view(-1, 1).to(self.device)
        self.batch_size = y.size(0)
        return X, y

    # 训练
    def fit_generator(self, data_generator, epochs=3, validation_data=None):
        self.valid_gen = validation_data
        self._best_metric = -np.Inf
        self._stopping_steps = 0
        self._total_batches = 0
        self._batches_per_epoch = len(data_generator)
        self._every_x_batches = int(np.ceil(self._batches_per_epoch))
        self._stop_training = False
        
        self.logger.info("Start training: {} batches/epoch".format(self._batches_per_epoch))
        for epoch in range(1, epochs+1):
            self.logger.info("************ Epoch={} start ************".format(epoch))
            epoch_loss = self._train_one_epoch(data_generator)
            self.logger.info("Train loss: {:.6f}".format(epoch_loss))
            if self._stop_training:
                break
            else:
                self.logger.info("************ Epoch={} end ************".format(epoch + 1))
        self.logger.info("Training finished.")

    # 一个epoch的训练过程
    def _train_one_epoch(self, data_generator):
        epoch_loss = 0
        self.train()
        batch_iterator = data_generator
        batch_iterator = tqdm(data_generator, disable=False, file=sys.stdout)
        for batch_index, batch_data in enumerate(batch_iterator):
            self.optimizer.zero_grad()
            return_dict = self.forward(batch_data)
            loss = self.loss_fn(return_dict["y_pred"], return_dict["y_true"])
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            self._on_batch_end(batch_index)
            if self._stop_training:
                break
        return epoch_loss / self._batches_per_epoch

    def _on_batch_end(self, batch):
        self._total_batches += 1
        if (batch + 1) % self._every_x_batches == 0 or (batch + 1) % self._batches_per_epoch == 0:
            epoch = round(float(self._total_batches) / self._batches_per_epoch, 2)
            val_logs, _, _ = self._evaluate_generator(self.valid_gen)
            self._checkpoint_and_earlystop(epoch, val_logs)
            self.logger.info("--- {}/{} batches finished ---".format(batch + 1, self._batches_per_epoch))

    def _checkpoint_and_earlystop(self, epoch, logs, min_delta=1e-6):
        monitor_value = logs['AUC']
        if monitor_value < self._best_metric - min_delta:
            self._stopping_steps += 1
            self.logger.info("Monitor STOP: {:.6f} !".format(monitor_value))
        else:
            self._stopping_steps = 0
            self._best_metric = monitor_value
            self.logger.info("Save best model: {:.6f}"\
                            .format(monitor_value))
            torch.save(self.state_dict(), self.checkpoint)
        if self._stopping_steps >= self._patience:
            self._stop_training = True
            self.logger.info("Early stopping at epoch={:g}".format(epoch))

    def _evaluate_generator(self, data_generator):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            y_true = []
            from tqdm import tqdm
            data_generator = tqdm(data_generator, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
                y_true.extend(return_dict["y_true"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
            y_true = np.array(y_true, np.float64)
            val_logs = self._evaluate_metrics(y_true, y_pred, self._validation_metrics)
        return val_logs, y_pred, y_true

    def _evaluate_metrics(self, y_true, y_pred, metrics):
        return evaluate_metrics(y_true, y_pred, metrics)

    def load_weights(self, checkpoint):
        # checkpoint: str 存放模型参数文件路径
        self.to(self.device)
        state_dict = torch.load(checkpoint, map_location="cpu")
        self.load_state_dict(state_dict)
        del state_dict
        torch.cuda.empty_cache()

    def count_parameters(self, count_embedding=True):
        total_params = 0
        for name, param in self.named_parameters(): 
            if not count_embedding and "embedding" in name:
                continue
            if param.requires_grad:
                total_params += param.numel()
        self.logger.info("Total number of parameters: {}.".format(total_params))
    
    def inference(self, test_data):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            from tqdm import tqdm
            data_generator = tqdm(test_data, disable=False, file=sys.stdout)
            for batch_data in data_generator:
                return_dict = self.forward(batch_data)
                y_pred.extend(return_dict["y_pred"].data.cpu().numpy().reshape(-1))
            y_pred = np.array(y_pred, np.float64)
        return y_pred

if __name__ == '__main__':
    # 配置日志相关文件
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)