# Copyright 2015 gRPC authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The Python implementation of the GRPC helloworld.Greeter server."""
# -*- coding: utf-8 -*-
from concurrent import futures
import logging
import os

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

import pickle
import hashlib
from Crypto.Cipher import AES
import time

import dataset_preprocess
import pandas as pd

machine_name = "server"
# AES Key
key = b'\xee\xae\x9a\xb0\x171*\x93L\xa3\x19\n\x91\xc1\x0b\xaf'
nonce = b'\xd81P\xdf\x98\x8e&UL\xb0\x10\x17'
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024 # 设置grpc最大可接受的文件大小 1G

class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def __init__(self) -> None:
        super().__init__()
        self.use_model = "default"
        self.hashtaskA = ""
        self.hashtaskB = ""
        # 定义散列算法为安全的SHA384
        self.sha3_hash = hashlib.sha384()
        self.receivedataA = False
        self.receivedataB = False
        self.passwd = "pass"
        self.key = key
        self.nonce = nonce
        self.cipher = AES.new(self.key, AES.MODE_GCM, self.nonce)
        self.idhashA = ""
        self.idhashB = ""
        self.readyA = False
        self.readyB = False

    def encryption(self,plaintext):
        return AES.new(self.key, AES.MODE_GCM, self.nonce).encrypt(plaintext)
    
    def decryption(self,ciphertext):
        return AES.new(self.key, AES.MODE_GCM, self.nonce).decrypt(ciphertext)

    def hashdigest(self,msg):
        hasher =  hashlib.sha384()
        hasher.update(msg)
        return hasher.hexdigest()

    # 将模型发送至ServerA
    def send_model_to_B(self, message, goal):
        runB(msg = message, gl = goal)

    # 检查machineA和machineB发送消息的合法性
    def checkB(self, message, goal):
        runB(msg = message, gl = goal)

    def checkA(self, message, goal):
        runA(msg = message, gl = goal)

    def checkTask(self):
        if(self.hashtaskA != "" and self.hashtaskB != ""):
            # 对通过比较确定使用的模型进行模型一致性检验
            hashmodel = self.hashdigest(self.use_model.encode('utf-8'))
            logger.info("HashModel is %s" % hashmodel)
            if (self.hashtaskA == self.hashtaskB):
                logger.info("Task negotiation succeed")
                logger.info("Passwd :" + self.passwd)
                hashver = self.hashdigest(self.passwd.encode('utf-8'))
                ciphVer = self.encryption(str.encode(self.passwd))
                logger.info("Ciphertext Ver: %s" % ciphVer)
                runA(msg = hashver, gl = "pass", stage = "train", ciph = ciphVer)
                runB(msg = hashver, gl = "pass", stage = "train", ciph = ciphVer)
            else:
                # 任务一致性验证不通过,协商错误
                logger.error("No pass for the consistency of task negotiation, error occurs")
                runA(msg = "task different", gl = "error", stage = "train")
                runB(msg = "task different", gl = "error", stage = "train")

    def checkDataA(self, msg):
        logger.info("Received dataset from A successfully, decryption starting...")
        datum = self.decryption(msg) # 获取解密数据
        # 将解密结果写入dataA.txt
        with open('./train_a_after_preprocess.csv', 'w') as f:
        # with open('./serverC/dataA.txt', 'w') as f:
            f.write(datum.decode())  
        hashA = self.hashdigest(datum)
        # 打开原始存储的datahashA.txt
        with open('./datahashA.txt', 'r') as f:
            datahash = f.read()
        # 验证散列结果是否相等
        logger.info(hashA)
        logger.info(datahash)
        if (hashA == datahash):
            self.receivedataA = True
            self.train_waiting()
        else:
            # 否则数据一致性错误
            runA(stage = "train", gl = "error", msg = "data_different")

    def checkDataB(self,msg):
        logger.info("Received dataset from B successfully, decryption starting...")
        datum = self.decryption(msg) # 获取解密数据  
        with open(r'train_b_after_preprocess.csv', 'w') as f:
            f.write(datum.decode()) 
        hashB = self.hashdigest(datum)
        with open('./datahashB.txt', 'r') as f:
            datahash = f.read()
        # 参与方B上传数据一致性验证成功
        logger.info(hashB)
        logger.info(datahash)
        if (hashB == datahash):
            self.receivedataB = True
            self.train_waiting()
        else:
            runB(stage = "train", gl = "error", msg = "data_different")

    def train_waiting(self):
        # 完成A, B两方数据接收
        if(self.receivedataA and self.receivedataB):
            logger.info("Training, please waiting")
            # 完成数据合并预处理
            train_A_after_preprocess = pd.read_csv(r'train_a_after_preprocess.csv')
            train_B_after_preprocess = pd.read_csv(r'train_b_after_preprocess.csv')
            # 数据合并
            all_data = pd.merge(train_A_after_preprocess, train_B_after_preprocess, left_on=['id'],
                            right_on=['id'], how='left')
            all_data = all_data.drop('id', axis=1)
            # 调整特征顺序
            feature_order = ['label', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'C1', 
                             'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'B1', 'B2', 'B3', 'B4', 
                             'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
                             'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
                             'D11', 'D12', 'D13']

            all_data = all_data[feature_order]
            all_data.to_csv('all_data.csv', index=False)
            # 训练集与验证集分割
            train_df, valid_df = dataset_preprocess.data_split(all_data)
            # 训练集与验证集保存
            train_df.to_csv('federal_train.csv', index=False)
            valid_df.to_csv('federal_valid.csv', index=False)
            
            logger.info("Dataset split finished")
            logger.info("Wait for traing..")
            # 进行训练
            train_command = "bash ./run_pytorch_on_occlum.sh &"
            os.system(train_command)
           
            logger.info("Finish train")
            # 将训练完成后的model发送给machineA
            logger.info("Sending the model file to machineA")
            with open('./occlum_instance/AFM.model', 'rb') as f:
               modelfile = f.read()
                
            runA(stage="train",msg="modelfile",ciph=modelfile,gl="modelfile")
            runB(stage="train",msg="modelfile",gl="modelfile")

    def checkIDHash(self):
        logger.info("hashing ")
        if(self.idhashA !="" and self.idhashB !=""):
            if(self.idhashA==self.idhashB):
                logger.info("hash equal")
                runA(msg="same",gl="predict",stage="predict")
                time.sleep(1)
                runB(msg="same",gl="predict",stage="predict")
            else:
                runA(msg="error id",gl="error",stage="predict")
                runA(msg="error id",gl="error",stage="predict")

    def getPredictData(self,name,cipher):
        cipher = self.decryption(cipher)
        logger.info("Receive the test data from participant[] " + name)
        with open(r'test_'+name+'.csv', 'wb') as f:
            f.write(cipher)
        logger.info("Get prediction data from A and B")
        self.predict_waiting()

    def predict_waiting(self):
        if(self.readyA and self.readyB):
            # 进行预测
            # 得到预测结果并进行加密
            logger.info("predicting")
            # 此处成功获取test_A.txt和test_B.txt
            # 进行数据合并与预测
            test_A_after_preprocess = pd.read_csv(r'test_A.csv')
            test_B_after_preprocess = pd.read_csv(r'test_B.csv')
            test_data = pd.merge(test_A_after_preprocess, test_B_after_preprocess, left_on=['id'],
                            right_on=['id'], how='left')
            test_data = test_data.drop('id', axis=1)
             # 调整特征顺序
            feature_order = ['label', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'C1', 
                             'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'B1', 'B2', 'B3', 'B4', 
                             'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13',
                             'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10',
                             'D11', 'D12', 'D13']
            
            test_data = test_data[feature_order]
            test_data.to_csv('federal_test.csv', index=False)
            logger.info("Test dataset received and merged successfully")
            
            # 执行预测
            predict_command = "bash ./run_predict_on_occlum.sh"
            os.system(predict_command)
            
            logger.info("Prediction finished")
            with open('./occlum_instance/result.csv', 'rb') as f:
                result= f.read()
            cipher = self.encryption(result)
            time.sleep(3)
            runA(msg=result.decode(),gl="result",stage="predict",ciph=cipher)
            # runB(msg=result.decode(),gl="result",stage="predict",ciph=cipher)
            logger.info('Prediction result sent')

    def SayHello(self, request, context):
        logger.info("|Request name: " + request.name + "|Request goal: " + request.goal)
        if(request.name == "machineA"):
            if(request.goal == "task1"):
                # 选定模型更改某些变量
                self.use_model = request.message
                logger.info("send model to B")
                self.send_model_to_B(message = request.message, goal = "task1")
                self.checkA(message = "check_task", goal = "check")
            elif(request.goal == "datahash"):
                with open ('./datahashA.txt', 'w') as f:
                    f.write(request.message)
                logger.info("datahash of A successfully received")
                time.sleep(3)
                self.send_model_to_B(message = request.message, goal = "datahashA")
                self.checkA(message = "check_hash", goal = "check")
        elif(request.name == "machineB"):
            if(request.goal == "datahash"):
                # 保存datahash
                with open('./datahashB.txt', 'w') as f:
                    f.write(request.message)
                logger.info("datahash of B successfully received")
                self.checkB(message = "check_hash", goal = "check")
                print("Finish negotiation")
                runA(msg="start", stage="train", gl="start")
                runB(msg="start", stage="train", gl="start")
        return helloworld_pb2.HelloReply(message=f'check, {machine_name}! goal is {request.goal} !')

    def SayHelloAgain(self, request, context):
        logger.info("|Request name: " + request.name + "|Request goal: " + request.goal)
        if(request.name == "machineA"):
            if(request.goal == "verify"):    
                self.hashtaskA = request.message
                self.checkTask()
            elif(request.goal == "senddata"):
                self.checkDataA(request.cipher)
        elif(request.name == "machineB"):
            if(request.goal == "verify"):
                self.hashtaskB = request.message
                self.checkTask()
            elif(request.goal == "senddata"):
                self.checkDataB(request.cipher)
        return helloworld_pb2.HelloReply(message = f'check, {machine_name}! goal is {request.goal} !') 
    
    def Predict(self, request, context):
        logger.info("|Request name: " + request.name + "|Request goal: " + request.goal)
        if(request.name == "machineA"):
            if(request.goal == "idhash"):    
                self.idhashA=request.message
                self.checkIDHash()
            elif(request.goal == "testdata"):
                self.readyA=True
                self.getPredictData(name='A',cipher=request.cipher)
        elif(request.name == "machineB"):
            if(request.goal == "idhash"):
                self.idhashB=request.message
                self.checkIDHash()
            elif(request.goal == "testdata"):
                self.readyB = True
                self.getPredictData(name='B',cipher=request.cipher)
        return helloworld_pb2.HelloReply(message = f'check, {machine_name}! goal is {request.goal} !') 

def runA(msg, gl, stage='', ciph = None):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with open('./cert2/client-key-A.pem', 'rb') as f:
        private_key = f.read()
    with open('./cert2/client-cert-A.pem', 'rb') as f:
        certificate_chain = f.read()
    with open('./cert2/ca-cert.pem', 'rb') as f:
        root_certificates = f.read()
    channel_creds = grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain)
    # with grpc.secure_channel('10.50.0.233:26900', channel_creds) as channel:
    with grpc.secure_channel('machineA:26900',channel_creds) as channel:
    # with grpc.insecure_channel('10.50.0.233:26900') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        # 训练过程
        if(stage == "train"):
            response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl, cipher = ciph))
        # 预测阶段
        elif(stage == "predict"):
            response = stub.Predict(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl, cipher= ciph))
        # 协商阶段
        else:
            if(gl == "model_list"):
                req = helloworld_pb2.HelloRequest(name = machine_name, goal = gl)
                req.models.extend(msg)
                response = stub.SayHello(req)
            else:
                response = stub.SayHello(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl))
        logger.info("Greeter server received: " + response.message)

def runB(msg, gl, stage='', ciph=None):
    with open('./cert2/client-key-B.pem', 'rb') as f:
        private_key = f.read()
    with open('./cert2/client-cert-B.pem', 'rb') as f:
        certificate_chain = f.read()
    with open('./cert2/ca-cert.pem', 'rb') as f:
        root_certificates = f.read()
    channel_creds = grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain)
    # with grpc.secure_channel('10.50.0.73:50053', channel_creds) as channel:
    with grpc.secure_channel('machineB:50053', channel_creds) as channel:
    # with grpc.insecure_channel('machineB:50053') as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        if(stage == "train"):
            response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl, cipher = ciph))
        elif(stage == "predict"):
            response = stub.Predict(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl, cipher= ciph))
        else:
            response = stub.SayHello(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl))
        logger.info("Greeter server received: " + response.message)

# Mutual TLS
def serve():
    port = '0.0.0.0:26901'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                  ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ])

    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    
    with open('./cert2/server-key.pem', 'rb') as f:
        private_key = f.read()
    with open('./cert2/server-cert.pem', 'rb') as f:
        certificate_chain = f.read()
    with open('./cert2/ca-cert.pem', 'rb') as f:
        root_certificates = f.read()
    server_creds = grpc.ssl_server_credentials(((private_key, certificate_chain),), root_certificates, True)
    server.add_secure_port( port, server_creds)
    # server.add_insecure_port(port)
    server.start()
    # 发送支持的模型列表
    with open('./serverC/models.txt', encoding='utf-8') as file_obj:
        mdls = []
        for line in file_obj:
            mdls.append(line)
    logger.info(mdls)
    # 向machineA发送支持的模型列表
    runA(msg = mdls, gl = "model_list")
    logger.info("Train start...")
    print("Server started, listening on " + port)
    server.wait_for_termination()
    
if __name__ == '__main__':
    # 配置日志相关文件
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    serve()
