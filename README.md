# 星河杯-基于可信执行环境的广告精准投放联合建模推理框架
该仓库用于存储"星河杯"隐私计算大赛广告精准投放赛道参赛项目的相关文件及开源代码；初步公开基于可信执行环境的联合建模推理框架的设计，实现方法及相关指标文档，完整代码于决赛结束后开源；

## 方案说明

完整方案说明见`基于可信执行环境的联合建模推理框架设计文档.md`；

## TaskInitiatorA

任务发起者相关代码，具体说明见`TaskInitiatorA/Architecture Description.md`；

## DataProviderB

数据提供方相关代码，具体说明见`DataProviderB/Architecture Description.md`；

## TaskPerformerC

联合训练/预测任务执行方相关代码，默认该方配备Intel SGX可信执行环境与Occlum库操作系统，具体说明见`TaskPerformerC/Architecture Description.md`；

## Data

训练及预测所需数据链接如下：https://pan.baidu.com/s/1QljKHtngQWNxj7NZOzeQzg，提取码: zbhd，包含以下内容：

* A方数据
  * train_a.txt
  * test_a.txt
* B方数据
  * train_b.txt
  * test_b.txt

