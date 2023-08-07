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
# -*- coding:utf-8 -*-
from concurrent import futures
import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc

import pickle
import hashlib
from Crypto.Cipher import AES
import time
import pandas as pd

import dataset_preprocess

machine_name = "machineA"
# AES Key
key = b'\xee\xae\x9a\xb0\x171*\x93L\xa3\x19\n\x91\xc1\x0b\xaf'
nonce = b'\xd81P\xdf\x98\x8e&UL\xb0\x10\x17'
MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

class Greeter(helloworld_pb2_grpc.GreeterServicer):
    def __init__(self) -> None:
        super().__init__()
        self.chosen_model = "default"
        # 待修改：采用SHA3
        self.sha3_hash = hashlib.sha384()
        # AES加密选择GCM模式
        self.key = key
        self.nonce = nonce

    def encryption(self,plaintext):
        return AES.new(self.key, AES.MODE_GCM, self.nonce).encrypt(plaintext)
    
    def decryption(self,ciphertext):
        return AES.new(self.key, AES.MODE_GCM, self.nonce).decrypt(ciphertext)
    
    def hashdigest(self,msg):
        hasher =  hashlib.sha384()
        hasher.update(msg)
        return hasher.hexdigest()

    def opt(self, mdls):
        # 打印model列表
        for model in mdls:
            logger.info("Model list: " + model)
        # 选择model
        self.chosen_model = input("please choose model you want:")
        run(msg = self.chosen_model, gl = "task1")

    def send_dataHash(self, message, goal):
        run(msg = message, gl = goal)

    def checkPass(self, ciph, msg):
        passwd = self.decryption(ciph)
        logger.info("Passwd: " + str(passwd))
        # 验证发送结果是否为pass
        passhash = self.hashdigest(passwd)
        logger.info(msg)
        logger.info(passhash)
        # 解密结果验证通过
        if(passhash == msg):
            logger.info("Verification Passed")
            with open(r'train_a_after_preprocess.csv', encoding='utf-8') as f:
                logger.info("Dataset A Uploading...")
                encrptdata = self.encryption(f.read().encode('utf-8'))
            # 上传加密结果
            # time.sleep(10)
            run(msg='send ciphertext',ciph = encrptdata, gl = "senddata", stage = "train")
        else:
            logger.error("No pass")
            run(msg = "pass_error", gl = "error", stage = "train")
        
    def SayHello(self, request, context):
        logger.info("|Request name: " + request.name + "|Request goal: " + request.goal)
        if(request.goal == "model_list"): # 收到C发的模型列表，选择模型
            self.opt(mdls = request.models)
        elif(request.goal == "check" and request.message == "check_task"):
            with open(r'train_a_after_preprocess.csv', 'rb') as f:
                datahashA = str(self.hashdigest(f.read()))
            logger.info(datahashA)
            logger.info("Send the hash of train data for A!")
            self.send_dataHash(message = datahashA, goal = "datahash")
        return helloworld_pb2.HelloReply(message=f'check, {machine_name}! goal is {request.goal} !')

    def SayHelloAgain(self, request, context):
        logger.info("|Request name: " + request.name + "|Request goal: " + request.goal)
        if(request.goal == "start"):
            hashmodel = self.hashdigest(self.chosen_model.encode('utf-8'))
            cipher = self.encryption(self.chosen_model.encode("utf-8"))
            time.sleep(1)
            run(msg = hashmodel, ciph = cipher, stage = "train", gl = "verify")
        elif(request.goal == "pass"):
            self.checkPass(request.cipher, request.message)
        elif(request.goal == "modelfile"):
            with open('./clientA/AFM.model','wb') as f:
                f.write(request.cipher)
            # 进行id hash  
            with open('./clientA/idlist.txt','rb') as f:
                idhash = self.hashdigest(f.read())
            run(msg=idhash,gl="idhash",stage="predict")
        return helloworld_pb2.HelloReply(message=f'check, {machine_name}! goal is {request.goal} !') 
    
    def Predict(self, request, context):    
        logger.info("|Request name: " + request.name + "|Request goal: " + request.goal)   
        # 预测阶段, 将预处理阶段生成的test_a_after_preprocess.csv文件发送至server
        if(request.goal == "predict"):
            with open (r'test_a_after_preprocess.csv', 'rb') as f:
                encrptdata = self.encryption(f.read())
            run(msg ="test data", ciph = encrptdata, stage = "predict", gl = "testdata")
        elif(request.goal == "result"):
            plainresult = self.decryption(request.cipher)
            if(self.hashdigest(plainresult) == self.hashdigest(str.encode(request.message))):
                logger.info("Integrity verification pass")
                with open ('./clientA/predictResult.csv', 'wb') as f:
                    f.write(request.message.encode('utf-8'))
                logger.info('Prediction finished')
            else:
                run(msg='result error',stage='predict',gl='error')
        return helloworld_pb2.HelloReply(message = f'check, {machine_name}! goal is {request.goal} !') 

def run(msg, gl, stage = '', ciph = None):
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with open('./cert2/client-key-C.pem', 'rb') as f:
        private_key = f.read()
    with open('./cert2/client-cert-C.pem', 'rb') as f:
        certificate_chain = f.read()
    with open('./cert2/ca-cert.pem', 'rb') as f:
        root_certificates = f.read()
    channel_creds = grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain)
    # with grpc.insecure_channel('10.16.116.7:26901') as channel:
    # with grpc.secure_channel('10.16.116.7:26901', channel_creds) as channel:
    with grpc.secure_channel('serverC:26901',channel_creds) as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        # 任务一, 若接收的stage任务为train
        if(stage == "train"):
            # logger.info(msg)
            # 调用SayHelloAgain函数
            response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl, cipher = ciph))
        # 任务二, 若接收的stage任务为predict
        elif(stage == "predict"):
            response = stub.Predict(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl,cipher= ciph))
        # 协商阶段
        else:
            response = stub.SayHello(helloworld_pb2.HelloRequest(name = machine_name, message = msg, goal = gl))
        logger.info("Greeter client received: " + response.message)
        
# Mutual TLS
def serve():
    # 接收50052端口发送的消息
    port = '0.0.0.0:26900'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                  ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH), ])

    helloworld_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    # 导入相关证书并完成安全信道建立过程
    with open('./cert2/server-key.pem', 'rb') as f:
        private_key = f.read()
    with open('./cert2/server-cert.pem', 'rb') as f:
        certificate_chain = f.read()
    with open('./cert2/ca-cert.pem', 'rb') as f:
        root_certificates = f.read()
    server_creds = grpc.ssl_server_credentials(((private_key, certificate_chain),), root_certificates, True)
    server.add_secure_port( port, server_creds)
    # 保持监听
    # server.add_insecure_port(port)
    server.start()
    logger.info("Server started, listening on " + port)
    server.wait_for_termination()


"""
参与方A作为数据提供方和结果接收方, 本地进行数据集预处理操作
预处理结果存储于train_a_after_preprocess.csv文件中
"""
def dataPreprocessMachinA():
    # 读取当前路径下的train_a.txt文件
    train_A = pd.read_csv(r'train_a.txt')
    test_A = pd.read_csv(r'test_a.txt')
    
    # 填补缺失值
    train_A = dataset_preprocess.fill_na(train_A)
    test_A = dataset_preprocess.fill_na(test_A)
    
    train_A['flag'] = 0
    test_A['flag'] = 1
    
    A = pd.concat([train_A, test_A])
    # 因子化处理
    A = dataset_preprocess.factorize(A)
    train_A = A[A['flag'] == 0]
    test_A = A[A['flag'] == 1]
    train_A = train_A.drop(columns=['flag'])
    test_A = test_A.drop(columns=['flag'])
    # 预处理后的数据导入train_a_after_preprocess.csv文件中
    train_A.to_csv(r'train_a_after_preprocess.csv', index=False)
    # 预处理后的测试数据导入test_a_after_preprocess.csv文件中
    test_A.to_csv(r'test_a_after_preprocess.csv', index=False)

if __name__ == '__main__':
    # 配置日志相关文件
    logging.basicConfig(level = logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # logger.info("Data preprocess process starts!")
    # dataPreprocessMachinA()
    # logger.info("Dataset split and preprocess finished!")
    logger.info("Data transmission...")
    serve()
