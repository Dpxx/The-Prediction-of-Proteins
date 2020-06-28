本文件夹为单实例方法：NASNet的代码，其中：

* processing.py为数据预处理，手动划分好训练集和验证集
* train.py为训练模型的代码
* test.py为测试验证集，计算三个评价指标的代码


* NASNet-mobile-no-top.h5为迁移学习预训练后的权重
* model.h5为训练过程中验证集正确率最高的模型


* train.log储存训练过程中每个epoch后训练集和验证集的acc和loss
* curve.py为对训练过程中的acc和loss变化输出图像的代码
