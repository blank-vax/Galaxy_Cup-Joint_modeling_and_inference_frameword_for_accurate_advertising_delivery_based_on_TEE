from __future__ import print_function

import logging

import grpc
import helloworld_pb2
import helloworld_pb2_grpc
import os

# runA(stage="train",msg="modelfile",ciph=modelfile,gl="modelfile")
# runB(stage="train",msg="modelfile",gl="modelfile")

def run():
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    # print("Will try to greet world ...")
    with open('./cert2/client-key.pem', 'rb') as f:
        private_key = f.read()
    with open('./cert2/client-cert.pem', 'rb') as f:
        certificate_chain = f.read()
    with open('./cert2/ca-cert.pem', 'rb') as f:
        root_certificates = f.read()
        channel_creds = grpc.ssl_channel_credentials(root_certificates, private_key, certificate_chain)
                                                                            
        logger.info("reading modelfile")
    train_command = "bash run_pytorch_on_occlum.sh"
    os.system(train_command)
    model_copy_command = "cp ./occlum_instance/AFM.model ./AFM.model"
    os.system(model_copy_command)
    with open('./occlum_instance/AFM.model', 'rb') as f:
        modelfile = f.read()
        logger.info("send model to MachineA")
    with grpc.insecure_channel("10.50.0.233:26900") as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name = "serverC", message = "modelfile", goal = "modelfile", cipher = modelfile))
        logger.info("Greeter client received: " + response.message)
        logger.info("send model to MachineB")
    with grpc.insecure_channel("10.50.0.73:50053") as channel:
        stub = helloworld_pb2_grpc.GreeterStub(channel)
        response = stub.SayHelloAgain(helloworld_pb2.HelloRequest(name = "serverC", message = "modelfile", goal = "modelfile"))
        logger.info("Greeter client received: " + response.message)
if __name__ == "__main__":
                                                                                                                                           logging.basicConfig()
                                                                                                                                           logger = logging.getLogger(__name__)
                                                                                                                                           run()
