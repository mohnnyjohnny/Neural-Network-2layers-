1、two-layer-mlp为代码文件，运行一次大概需要8-10min。在跑第一个epoch时由于牵涉到画图保存数据点，因此速度较慢；
2、只需要运行two-layer-mlp即可完成训练和测试；
3、运行two-layer-mlp会输出最终的test acc，基本稳定在98%以上；同时会输出四张图，分别为loss曲线（包含训练集和验证集）、测试集的accuracy曲线、w1矩阵热力图、w2矩阵热力图；
4、mnist_data文件中含有四个mnist数据集和一个weight.npy文件。weight.npy是模型输出的W和b，每次运行two-layer-mlp都会更新w和b，也会更新保存weight.npy文件。
